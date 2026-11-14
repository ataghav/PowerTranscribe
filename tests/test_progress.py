from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from progress import Progress
import transcribe


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
    def test_terminal_updates_same_row_and_finishes_once(self):
        output = Terminal()
        progress = Progress(output)
        progress.section('processing')
        progress.start('step recognition')
        progress.update(.25)
        progress.last = 0
        progress.update(.8)
        progress.finish()
        self.assertIn('\r\033[2K  step recognition: progress 25%', output.getvalue())
        self.assertIn('\r\033[2K  step recognition: progress 80%', output.getvalue())
        self.assertTrue(output.getvalue().endswith('step recognition: done\n'))
        self.assertEqual(output.getvalue().count('\n'), 4)  # heading plus one completed row

    def test_redirected_output_has_no_updates_or_escape_codes(self):
        output = io.StringIO()
        progress = Progress(output)
        progress.start('model example', 'checking locally')
        for i in range(100):
            progress.update(i/100)
        progress.finish('detected locally')
        self.assertEqual(output.getvalue(), '  model example: checking locally\n  model example: detected locally\n')

    def test_failed_stage_does_not_report_done(self):
        output = Terminal()
        with patch.object(transcribe, 'PROGRESS', Progress(output)), patch.object(transcribe, 'MONITOR', None), \
                patch.object(transcribe, 'gpu_memory', return_value='test'):
            with self.assertRaisesRegex(ValueError, 'failure'):
                with transcribe.stage('test', show_progress=True):
                    raise ValueError('failure')
        self.assertIn('step test: failed', output.getvalue())
        self.assertNotIn('done', output.getvalue())

    def test_job_shows_model_steps_and_summary_without_transcript(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'audio.wav'
            source.touch()
            terminal, transcript = Terminal(), io.StringIO()
            resources = Mock()
            resources.preflight.return_value = {}
            resources.snapshot.return_value = {}

            def local_model(repo, progress=None):
                if progress:
                    progress('detected locally', 1)
                return root

            def detect(audio, store, config, monitor):
                store.add('turns', [dict(start=0., end=1., speaker='SPEAKER_01')])

            def align(audio, store, config, monitor):
                store.add('words', [dict(start=0., end=1., text='not printed')])

            handlers = list(transcribe.LOGGER.handlers)
            try:
                with patch('sys.argv', ['transcribe.py', str(source), '--out', str(root/'run')]), \
                        patch.object(transcribe, 'configure_logging'), \
                        patch.object(transcribe, 'probe_duration', return_value=1.), \
                        patch.object(transcribe, 'Resources', return_value=resources), \
                        patch.object(transcribe, 'ensure_model', side_effect=local_model), \
                        patch.object(transcribe, 'decode_to_disk', return_value=Mock()), \
                        patch.object(transcribe, 'diarize', side_effect=detect), \
                        patch.object(transcribe, 'recognize'), \
                        patch.object(transcribe, 'align', side_effect=align), \
                        patch.object(transcribe, 'gpu_memory', return_value='test'), \
                        patch.object(transcribe.torch.cuda, 'get_device_properties', return_value='test'), \
                        patch.object(transcribe, 'MONITOR', None), \
                        redirect_stderr(terminal), redirect_stdout(transcript):
                    transcribe.main()
                output = terminal.getvalue()
                self.assertEqual(output.count('detected locally'), 3)
                self.assertEqual(output.count(': done'), 6)
                self.assertIn('finished successfully!', output)
                self.assertIn('# speakers: 1', output)
                self.assertIn('# speaking turns: 1', output)
                self.assertEqual(transcript.getvalue(), '')
            finally:
                for handler in transcribe.LOGGER.handlers:
                    if handler not in handlers:
                        handler.close()
                transcribe.LOGGER.handlers = handlers
