import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from runtime import GIB, ResourceLimitError, Resources, Settings
from storage import AudioFile, Checkpoints, silence_boundary
import transcribe

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = Settings.read(ROOT/'transcribe.toml')

    def audio(self, samples):
        path = self.directory/'audio.f32'
        np.asarray(samples, dtype='<f4').tofile(path)
        return AudioFile(path, self.config.max_chunk_seconds+2*self.config.context_seconds)

    def test_last_silence_before_limit_and_exact_coverage(self):
        wave = np.ones(120*16000, dtype=np.float32)*0.1
        wave[37*16000:38*16000] = 0
        audio = self.audio(wave)
        boundary = silence_boundary(audio, 0, 40, self.config)
        self.assertTrue(37 <= boundary <= 38)
        start, cores = 0., []
        while start < audio.duration:
            limit = min(audio.duration, start+self.config.max_chunk_seconds)
            end = limit if limit == audio.duration else silence_boundary(audio, start, limit, self.config)
            self.assertGreater(end, start)
            self.assertLessEqual(end-start, self.config.max_chunk_seconds)
            cores.append(audio.read(start, end))
            start = end
        np.testing.assert_array_equal(np.concatenate(cores), wave)

    def test_continuous_speech_and_short_tail(self):
        audio = self.audio(np.ones(41*16000+123, dtype=np.float32))
        cut = silence_boundary(audio, 0, 40, self.config)
        self.assertTrue(35 <= cut <= 40)
        self.assertEqual(len(audio.read(cut, audio.duration)), audio.samples-round(cut*16000))
        with self.assertRaisesRegex(ValueError, 'beyond-budget'):
            audio.read_samples(0, audio.max_read_samples+1)

    def test_large_file_does_not_require_large_read(self):
        path = self.directory/'large.f32'
        with path.open('wb') as file:
            file.truncate(24*3600*16000*4)  # sparse 24-hour PCM file
        audio = AudioFile(path, 42)
        self.assertEqual(audio.duration, 24*3600)
        self.assertEqual(len(audio.read(86399, 86400)), 16000)
        self.assertLessEqual(audio.read(0, 40).nbytes, 40*16000*4)

    def test_config_rejects_unknown_nonfinite_and_inconsistent_values(self):
        original = (ROOT/'transcribe.toml').read_text()
        for text, expected in [
            (original+'\ntypo_limit = 1\n', 'unknown'),
            (original.replace('ram_budget_gib = 12.0', 'ram_budget_gib = nan'), 'ram_budget_gib'),
            (original.replace('max_new_tokens = 2048', 'max_new_tokens = true'), 'max_new_tokens'),
            (original.replace('min_chunk_seconds = 10.0', 'min_chunk_seconds = 100.0'), 'min_chunk_seconds'),
            (original.replace('max_chunk_seconds = 40.0', 'max_chunk_seconds = 180.0'), '180-second'),
        ]:
            path = self.directory/'invalid.toml'
            path.write_text(text)
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                Settings.read(path)

    def test_preflight_rejects_budgets_before_applying_gpu_cap(self):
        gpu = Mock()
        gpu.version.hip = 'test'
        gpu.cuda.is_available.return_value = True
        gpu.cuda.mem_get_info.return_value = (16*GIB, 16*GIB)
        for cfg, match in [
            (dataclasses.replace(self.config, ram_budget_gib=1), 'too small'),
            (dataclasses.replace(self.config, gpu_budget_gib=100), 'Insufficient GPU'),
            (dataclasses.replace(self.config, working_disk_gib=1000), 'Insufficient working disk'),
        ]:
            with patch('runtime.host_memory', return_value=(32*GIB, 24*GIB)), \
                    patch('runtime.shutil.disk_usage', return_value=Mock(free=100*GIB)):
                with self.assertRaisesRegex(ResourceLimitError, match):
                    Resources(cfg, self.directory, gpu).preflight(90)
        gpu.cuda.set_per_process_memory_fraction.assert_not_called()

    def test_disk_budget_stops_further_writes(self):
        resources = Resources(dataclasses.replace(self.config, working_disk_gib=0.01), self.directory, Mock())
        with self.assertRaisesRegex(ResourceLimitError, 'budget exceeded'):
            resources.check_disk(20*1024**2)

    def test_export_keeps_words_with_identical_timestamps_distinct(self):
        store = Checkpoints(self.directory/'checkpoints.sqlite3', self.config)
        self.addCleanup(store.close)
        store.add('turns', [dict(start=0., end=2., speaker='SPEAKER_01')])
        store.add('words', [dict(start=0., end=0.5, text=text) for text in ('Hello', 'world')])
        resources = Mock()
        transcribe.export(self.directory, store, Path('source.wav'), self.config, resources)
        result = json.loads((self.directory/'transcript.json').read_text())
        self.assertEqual([w['text'] for w in result['words']], ['Hello', 'world'])
        self.assertTrue(all(w['speaker'] == 'SPEAKER_01' for w in result['words']))
        self.assertEqual(result['segments'][0]['text'], 'Hello world')
        self.assertFalse(list(self.directory.glob('*.part')))

    def test_checkpoint_survives_reopen(self):
        path = self.directory/'checkpoint.sqlite3'
        store = Checkpoints(path, self.config)
        value = dict(start=0., end=1., core_start=0., core_end=1., text='retained')
        store.add('chunks', [value])
        store.close()
        import sqlite3
        with sqlite3.connect(path) as connection:
            self.assertEqual(json.loads(connection.execute('SELECT data FROM chunks').fetchone()[0]), value)

    def test_rerun_preserves_previous_output_and_reserves_fresh_directory(self):
        requested = self.directory/'panel.transcript'
        first = transcribe.select_output_directory(requested, reserve=True)
        (first/'transcript.txt').write_text('previous results')
        planned = transcribe.select_output_directory(requested)
        self.assertEqual(planned.name, 'panel-2.transcript')
        self.assertFalse(planned.exists())
        second = transcribe.select_output_directory(requested, reserve=True)
        third = transcribe.select_output_directory(requested, reserve=True)
        self.assertEqual(second.name, 'panel-2.transcript')
        self.assertEqual(third.name, 'panel-3.transcript')
        self.assertEqual((first/'transcript.txt').read_text(), 'previous results')

    def test_output_file_is_rejected(self):
        path = self.directory/'result'
        path.write_text('keep this file')
        with self.assertRaisesRegex(ValueError, 'not a directory'):
            transcribe.select_output_directory(path, reserve=True)
        self.assertEqual(path.read_text(), 'keep this file')

    def test_attribution_sweep_retains_future_turns_for_nested_words(self):
        turns = [dict(start=0., end=.5, speaker='SPEAKER_01'),
                 dict(start=.3, end=.8, speaker='SPEAKER_02'),
                 dict(start=1.5, end=2., speaker='SPEAKER_01')]
        words = [dict(start=.1, end=1.8, text='long'), dict(start=.2, end=.21, text='short'),
                 dict(start=.6, end=.7, text='middle'), dict(start=1.6, end=1.7, text='later')]
        expected = [dict(word) for word in words]
        transcribe.attribute(expected, turns)
        store = Checkpoints(self.directory/'checkpoints.sqlite3', self.config)
        self.addCleanup(store.close)
        store.add('turns', turns)
        store.add('words', words)
        transcribe.export(self.directory, store, Path('source.wav'), self.config, Mock())
        actual = json.loads((self.directory/'transcript.json').read_text())['words']
        self.assertEqual(actual, expected)

    def test_recognition_retry_is_bounded_and_preserves_core_coverage(self):
        class Inputs(dict):
            def to(self, *args):
                return self

        class Processor:
            def apply_transcription_request(self, audio, language):
                return Inputs(input_ids=torch.ones((1, 1), dtype=torch.long), seconds=len(audio)/16000)

            def decode(self, ids, return_format):
                return ['recognized']

        model = Mock()
        model.generate.side_effect = lambda **kw: torch.zeros(
            (1, 2049 if kw['seconds'] > 22 else 3), dtype=torch.long)
        audio = self.audio(np.ones(80*16000, dtype=np.float32))
        store = Checkpoints(self.directory/'checkpoints.sqlite3', self.config)
        self.addCleanup(store.close)
        before = torch.backends.miopen.immediate
        with patch.object(transcribe, 'load', return_value=(Processor(), model)), \
                patch.object(transcribe, 'release'), \
                patch.object(transcribe, 'silence_boundary', side_effect=lambda audio, a, b, cfg: b):
            transcribe.recognize(audio, store, self.config, Mock())
        chunks = list(store.rows('chunks'))
        self.assertEqual([(c['core_start'], c['core_end']) for c in chunks],
                         [(0., 20.), (20., 40.), (40., 60.), (60., 80.)])
        self.assertEqual(torch.backends.miopen.immediate, before)
        model.generate.side_effect = lambda **kw: torch.zeros((1, 2049), dtype=torch.long)
        model.generate.reset_mock()
        with patch.object(transcribe, 'load', return_value=(Processor(), model)), \
                patch.object(transcribe, 'release'), \
                patch.object(transcribe, 'silence_boundary', side_effect=lambda audio, a, b, cfg: b):
            with self.assertRaisesRegex(RuntimeError, 'bounded retries'):
                transcribe.recognize(audio, store, self.config, Mock())
        self.assertEqual(model.generate.call_count, 3)
        self.assertEqual(store.count('chunks'), 4)  # earlier committed work remains


if __name__ == '__main__':
    unittest.main()
