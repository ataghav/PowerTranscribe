#!/usr/bin/env python3
"""uv run python transcribe.py meeting.wav [--out result]"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from runtime import Settings, Resources, ResourceLimitError, probe_duration
from storage import Checkpoints, decode_to_disk, silence_boundary, export_array, write_json_array
from model_store import REVISIONS, ensure_model
import torch
from transformers import (
    AutoProcessor, AutoModelForMultimodalLM,
    AutoModelForTokenClassification, AutoModelForAudioFrameClassification,
)

SR = 16000
DEVICE = "cuda:0"  # PyTorch uses the cuda namespace for AMD ROCm too.
ASR = "Qwen/Qwen3-ASR-1.7B-hf"
ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
DIARIZER = "nvidia/Nemotron-3-Diarization"
LOGGER = logging.getLogger("transcribe")
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR/'input'
OUTPUT_DIR = PROJECT_DIR/'output'
DEBUG = False
MONITOR = None


def log(message, console=False):
    LOGGER.log(logging.INFO if console else logging.DEBUG, message)


class ConsoleFormatter(logging.Formatter):
    def __init__(self, debug):
        super().__init__('%(asctime)s %(levelname)s %(message)s')
        self.debug = debug

    def format(self, record):
        # Keep full tracebacks in run.log; normal console errors stay concise.
        return super().format(record) if self.debug else record.getMessage()


def configure_logging(debug):
    from transformers.utils import logging as transformers_logging

    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        if handler.name == 'transcribe-console':
            LOGGER.removeHandler(handler)
            handler.close()
    console = logging.StreamHandler()
    console.name = 'transcribe-console'
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(ConsoleFormatter(debug))
    LOGGER.addHandler(console)
    if debug:
        transformers_logging.enable_progress_bar()
    else:
        transformers_logging.disable_progress_bar()


def gpu_memory():
    if MONITOR is not None:
        return MONITOR.describe()
    if not torch.cuda.is_available():
        return "GPU unavailable"
    try:
        free, total = torch.cuda.mem_get_info()
        gib = 1024 ** 3
        return (f"GPU GiB: allocated={torch.cuda.memory_allocated() / gib:.2f}, "
                f"reserved={torch.cuda.memory_reserved() / gib:.2f}, "
                f"peak={torch.cuda.max_memory_allocated() / gib:.2f}, "
                f"free={free / gib:.2f}/{total / gib:.2f}")
    except Exception as error:
        return f"GPU memory query unavailable: {error}"


@contextmanager
def stage(name):
    started = time.monotonic()
    log(f"Starting {name}; {gpu_memory()}")
    try:
        if MONITOR is not None:
            MONITOR.check()
        yield
        if MONITOR is not None:
            MONITOR.check()
        if DEBUG and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        LOGGER.debug("Failed %s after %.1fs; %s", name, time.monotonic() - started, gpu_memory())
        raise
    else:
        log(f"Finished {name} in {time.monotonic() - started:.1f}s; {gpu_memory()}")


def log_inputs(inputs):
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            log(f"Input {name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load(repo, model_class, dtype=torch.float16):
    with stage(f"loading {repo} revision={REVISIONS[repo]} dtype={dtype}"):
        local = str(ensure_model(repo))
        processor = AutoProcessor.from_pretrained(local, local_files_only=True)
        model = model_class.from_pretrained(
            local, local_files_only=True, dtype=dtype, device_map={"": DEVICE},
        ).eval()
    return processor, model


def release():
    gc.collect()
    torch.cuda.empty_cache()


def diarize(audio, store, config, resources):
    # The feature extractor counts floor(samples / hop_length) valid frames.
    processor, model = load(DIARIZER, AutoModelForAudioFrameClassification, torch.float32)
    processor.set_streaming_mode(config.diarization_mode)
    audio.max_read_samples = max(audio.max_read_samples, processor.num_samples_first_audio_chunk,
                                 processor.num_samples_per_audio_chunk)
    cache, previous, opened = None, np.zeros(model.config.head_config.num_speakers, dtype=np.int8), {}
    frame_cursor, first, next_report = 0, True, 0.
    frame_seconds = processor.feature_extractor.hop_length / SR
    total_frames = audio.samples // processor.feature_extractor.hop_length
    if not total_frames:
        del model, processor
        release()
        return
    with torch.inference_mode(), torch.backends.miopen.flags(immediate=config.miopen_immediate):
        while True:
            resources.check()
            start = 0 if first else processor.audio_chunk_start(frame_cursor)
            size = processor.num_samples_first_audio_chunk if first else processor.num_samples_per_audio_chunk
            end = min(audio.samples, start+size)
            last = end == audio.samples
            clip = audio.read_samples(start, end-start)
            if first and last and len(clip) <= processor.feature_extractor.n_fft // 2:
                clip = np.pad(clip, (0, processor.feature_extractor.n_fft-len(clip)))
            inputs = processor(clip, sampling_rate=SR, is_streaming=True,
                               is_first_audio_chunk=first, is_last_audio_chunk=last).to(DEVICE, torch.float32)
            if DEBUG and first:
                log_inputs(inputs)
            # A first-and-last chunk must also select the model's streaming path.
            lookahead = inputs.pop('num_lookahead_frames', 0)
            output = model(**inputs, speaker_cache=cache, num_lookahead_frames=lookahead)
            cache = output.speaker_cache
            active = (output.logits[0].sigmoid() > config.speaker_threshold).cpu().numpy().astype(np.int8)
            active = active[:total_frames-frame_cursor]
            changes = np.diff(np.concatenate([previous[None, :], active]), axis=0)
            turns = []
            for index, speaker in zip(*np.nonzero(changes)):
                time = min(audio.duration, (frame_cursor+int(index))*frame_seconds)
                label = f'SPEAKER_{int(speaker)+1:02d}'
                if changes[index, speaker] > 0:
                    opened[label] = time
                else:
                    begin = opened.pop(label)
                    if time > begin:
                        turns.append(dict(start=begin, end=time, speaker=label))
            if turns:
                store.add('turns', turns)
            frame_cursor += len(active)
            if len(active):
                previous = active[-1]
            if frame_cursor*frame_seconds >= next_report or last:
                log(f'1/3 Speaker detection: {min(audio.duration, frame_cursor*frame_seconds):.1f}/{audio.duration:.1f}s; '
                    f'{resources.describe()}')
                next_report = frame_cursor*frame_seconds + config.max_chunk_seconds
            del inputs, output, active, changes
            first = False
            if last:
                break
    if frame_cursor != total_frames:
        raise RuntimeError(f'Diarization frame coverage mismatch: {frame_cursor} != {total_frames}')
    store.add('turns', (dict(start=start, end=audio.duration, speaker=speaker)
                        for speaker, start in opened.items() if start < audio.duration))
    del model, processor, cache
    release()


def recognize(audio, store, config, resources):
    processor, model = load(ASR, AutoModelForMultimodalLM)

    def recognize_range(core_start, core_end):
        resources.check()
        start, end = max(0., core_start-config.context_seconds), min(audio.duration, core_end+config.context_seconds)
        clip = audio.read(start, end)
        text = ''
        if np.any(clip):
            # The author's supported minimum input is 0.5 s; keep original timestamps.
            if len(clip) < SR//2:
                clip = np.pad(clip, (0, SR//2-len(clip)))
            inputs = processor.apply_transcription_request(audio=clip, language='English').to(DEVICE, torch.float16)
            with torch.inference_mode(), torch.backends.miopen.flags(immediate=config.miopen_immediate):
                output = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=False, num_beams=1)
            ids = output[:, inputs['input_ids'].shape[1]:]
            reached_limit = ids.shape[1] >= config.max_new_tokens
            if not reached_limit:
                text = processor.decode(ids, return_format='transcription_only')[0].strip()
            del inputs, output, ids
            if reached_limit:
                if core_end-core_start <= config.min_chunk_seconds:
                    raise RuntimeError(f'Recognizer reached max_new_tokens={config.max_new_tokens} at '
                                       f'{core_start:.1f}–{core_end:.1f}s after bounded retries; '
                                       'refusing truncated output. Completed chunks are in checkpoints.sqlite3.')
                midpoint = silence_boundary(audio, core_start, (core_start+core_end)/2, config)
                log(f'Recognizer token limit at {core_start:.1f}–{core_end:.1f}s; retrying split at {midpoint:.2f}s')
                recognize_range(core_start, midpoint)
                recognize_range(midpoint, core_end)
                return
        store.add('chunks', [dict(start=start, end=end, core_start=core_start, core_end=core_end, text=text)])
        resources.check()

    core_start = 0.
    while core_start < audio.duration:
        limit = min(audio.duration, core_start+config.max_chunk_seconds)
        core_end = limit if limit == audio.duration else silence_boundary(audio, core_start, limit, config)
        log(f'2/3 Recognizing words: {core_start:.2f}–{core_end:.2f}s / {audio.duration:.2f}s; {resources.describe()}')
        recognize_range(core_start, core_end)
        core_start = core_end
    del model, processor
    release()


def align(audio, store, config, resources):
    processor, model = load(ALIGNER, AutoModelForTokenClassification)
    count = store.count('chunks')
    for i, chunk in enumerate(store.rows('chunks'), 1):
        resources.check()
        log(f'3/3 Aligning words: {i}/{count}; {resources.describe()}')
        if not chunk['text']:
            continue
        clip = audio.read(chunk['start'], chunk['end'])
        inputs, word_lists = processor.prepare_forced_aligner_inputs(
            audio=clip, transcript=chunk['text'], language='English')
        inputs = inputs.to(DEVICE, torch.float16)
        if DEBUG and i == 1:
            log_inputs(inputs)
        with torch.inference_mode(), torch.backends.miopen.flags(immediate=config.miopen_immediate):
            output = model(**inputs)
        aligned = processor.decode_forced_alignment(
            logits=output.logits, input_ids=inputs['input_ids'], word_lists=word_lists,
            timestamp_token_id=model.config.timestamp_token_id)[0]
        if not aligned:
            raise RuntimeError(f'Alignment returned no words at {chunk["core_start"]:.1f}s; recognition is checkpointed')
        words = []
        for word in aligned:
            a, b = float(word['start_time']), float(word['end_time'])
            if not math.isfinite(a+b) or b < a:
                raise ValueError('Invalid alignment timestamp')
            a = min(len(clip)/SR, max(0., a)) + chunk['start']
            b = min(len(clip)/SR, max(0., b)) + chunk['start']
            if chunk['core_start'] <= (a+b)/2 < chunk['core_end']:
                words.append(dict(start=a, end=b, text=word['text']))
        store.add('words', words)
        del inputs, output, aligned, words
    del processor, model
    release()


def activity_regions(turns):
    events = defaultdict(list)
    for turn in turns:
        events[turn["start"]].append((turn["speaker"], 1))
        events[turn["end"]].append((turn["speaker"], -1))
    active, regions, previous = defaultdict(int), [], None
    for time in sorted(events):
        speakers = sorted(s for s, count in active.items() if count > 0)
        if previous is not None and speakers and time > previous:
            regions.append((previous, time, speakers))
        for speaker, delta in events[time]:
            active[speaker] += delta
        previous = time
    return regions


def attribute(words, turns):
    regions, cursor = activity_regions(turns), 0
    for word in words:
        start, end = word["start"], max(word["end"], word["start"] + .005)
        scores, overlap = defaultdict(float), 0.
        while cursor < len(regions) and regions[cursor][1] <= start:
            cursor += 1
        for j in range(cursor, len(regions)):
            a, b, speakers = regions[j]
            if a >= end:
                break
            intersection = max(0., min(b, end) - max(a, start))
            for speaker in speakers:
                scores[speaker] += intersection
            if len(speakers) > 1:
                overlap += intersection
        ranked = sorted(scores, key=lambda s: -scores[s])
        label = ranked[0] if ranked else "UNKNOWN"
        if not ranked or scores[ranked[0]] < .5 * (end - start):
            label = "UNKNOWN"
        elif overlap > .02:
            label = "OVERLAP[" + ",".join(ranked) + "]"
        elif len(ranked) > 1 and scores[ranked[1]] >= .35 * scores[ranked[0]]:
            label = "UNCERTAIN[" + ",".join(ranked) + "]"
        word.update(speaker=label, overlapping=overlap > .02, candidates=ranked)


def timestamp(seconds, srt=False):
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}{"," if srt else "."}{milliseconds:03}'


def attributed_words(store, resources):
    """Attribute the ordered word stream without accumulating the recording."""
    turns = iter(store.rows('turns'))
    upcoming = next(turns, None)
    active = []
    for identity, word in store.words_with_ids():
        resources.check()
        end = max(word['end'], word['start']+.005)
        active = [turn for turn in active if turn['end'] > word['start']]
        while upcoming is not None and upcoming['start'] < end:
            if upcoming['end'] > word['start']:
                active.append(upcoming)
            upcoming = next(turns, None)
        # Sweep both ordered streams once; retained turns span at most one word's look-ahead.
        attribute([word], [turn for turn in active if turn['start'] < end])
        # Re-export always derives labels from diarization, not earlier inference.
        word.pop('original_speaker', None)
        word.pop('speaker_inference', None)
        yield identity, word


def fill_unknown_speakers(words, max_words=2, max_gap_seconds=0.5):
    """Bounded look-ahead; never bridge differing speakers or conflicting evidence."""
    pending, previous, oversized = [], None, False
    for identity, word in words:
        if word['speaker'] == 'UNKNOWN' and max_words > 0 and max_gap_seconds > 0:
            if oversized:
                yield identity, word
            else:
                pending.append((identity, word))
                if len(pending) > max_words:
                    yield from pending
                    pending.clear()
                    oversized = True
            continue
        if pending:
            speaker = word['speaker']
            chain = ([previous] if previous is not None else []) + [w for _, w in pending] + [word]
            matches = (previous is not None and speaker.startswith('SPEAKER_')
                       and previous['speaker'] == speaker
                       and not previous.get('overlapping', False) and not word.get('overlapping', False))
            close = all(-1e-9 <= b['start']-a['end'] <= max_gap_seconds+1e-9
                        for a, b in zip(chain, chain[1:]))
            compatible = all(not w.get('overlapping', False)
                             and all(candidate == speaker for candidate in w.get('candidates', []))
                             for _, w in pending)
            if matches and close and compatible:
                for _, unknown in pending:
                    unknown.update(original_speaker='UNKNOWN', speaker=speaker,
                                   speaker_inference='matching_neighbors')
            yield from pending
            pending.clear()
        yield identity, word
        previous, oversized = word, False
    yield from pending


def export(out, store, source, resources, include_timestamps=True,
           unknown_max_words=2, unknown_max_gap_seconds=0.5):
    store.prepare_export()
    segment, segment_id = None, None
    words = fill_unknown_speakers(attributed_words(store, resources),
                                 unknown_max_words, unknown_max_gap_seconds)
    for identity, word in words:
        if segment and segment['speaker'] == word['speaker']:
            segment['end'] = max(segment['end'], word['end'])
        else:
            if segment is not None:
                store.finish_segment(segment_id, segment)
            segment = {key: word[key] for key in ('start', 'end', 'speaker')}
            segment_id = store.start_segment(segment)
        store.update_word(identity, word, segment_id)
    if segment is not None:
        store.finish_segment(segment_id, segment)
    with (out/'transcript.txt.part').open('w', encoding='utf-8') as text_file, \
            (out/'transcript.srt.part').open('w', encoding='utf-8') as srt_file:
        for i, (identity, segment) in enumerate(store.segments_with_ids(), 1):
            prefix = (f'[{timestamp(segment["start"])}–{timestamp(segment["end"])}] '
                      if include_timestamps else '') + segment['speaker']+': '
            text_file.write(prefix)
            print(prefix, end='')
            srt_file.write(f'{i}\n{timestamp(segment["start"], True)} --> '
                           f'{timestamp(max(segment["end"], segment["start"]+.05), True)}\n'
                           f'{segment["speaker"]}: ')
            first = True
            for word in store.segment_words(identity):
                fragment = ('' if first else ' ')+word['text']
                resources.check_disk(len(fragment.encode('utf-8'))*2)
                text_file.write(fragment)
                srt_file.write(fragment)
                print(fragment, end='')
                first = False
            text_file.write('\n')
            srt_file.write('\n\n')
            print()
    with (out/'transcript.json.part').open('w', encoding='utf-8') as file:
        file.write('{"source":'+json.dumps(str(source))+',"models":'+json.dumps(REVISIONS))
        file.write(',"words":')
        write_json_array(file, store.rows('words'), resources)
        file.write(',"segments":[')
        first_segment = True
        for identity, segment in store.segments_with_ids():
            file.write(('' if first_segment else ',')+json.dumps(segment)[:-1]+',"text":"')
            first_word = True
            for word in store.segment_words(identity):
                fragment = ('' if first_word else ' ')+json.dumps(word['text'], ensure_ascii=False)[1:-1]
                file.flush()
                resources.check_disk(len(fragment.encode('utf-8')))
                file.write(fragment)
                first_word = False
            file.write('"}')
            first_segment = False
        file.write('],"diarization":')
        write_json_array(file, store.rows('turns'), resources)
        file.write('}')
    for name in ('transcript.txt', 'transcript.srt', 'transcript.json'):
        (out/(name+'.part')).replace(out/name)


def select_output_directory(requested, reserve=False):
    """Use a fresh sibling for reruns; mkdir reserves it atomically."""
    if not requested.name:
        raise ValueError('Output must name a run directory, not a filesystem root')
    if requested.exists() and not requested.is_dir():
        raise ValueError(f'Output path is not a directory: {requested}')
    parent = requested.parent
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError(f'Output parent is not a directory: {parent}')
    number = 1
    while True:
        if number == 1:
            candidate = requested
        elif requested.suffix == '.transcript':
            candidate = requested.with_name(f'{requested.stem}-{number}.transcript')
        else:
            candidate = requested.with_name(f'{requested.name}-{number}')
        if reserve:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                if not (candidate.exists() or candidate.is_symlink()):
                    raise  # A parent collided; trying more sibling names cannot resolve it.
        elif not candidate.exists():
            return candidate
        number += 1


def resolve_audio_path(path):
    path = path.expanduser()
    if path.is_absolute():
        candidates = [path]
    elif path.parent == Path('.'):
        candidates = [INPUT_DIR/path, path]
    else:
        candidates = [path, PROJECT_DIR/path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise FileNotFoundError(f'Audio file not found: {path}. Put recordings in {INPUT_DIR} or pass an explicit path.')


def requested_output_path(option, source):
    prefix = datetime.now().strftime('%y%d%m-%H%M__')
    if option is None:
        return OUTPUT_DIR/(prefix+source.stem+'.transcript')
    option = option.expanduser()
    if option.is_absolute():
        candidate = option.resolve()
    elif len(option.parts) > 1 and option.parts[0] == 'output':
        candidate = (PROJECT_DIR/option).resolve()
    else:
        candidate = (OUTPUT_DIR/option).resolve()
    if not option.is_absolute() and not candidate.is_relative_to(OUTPUT_DIR.resolve()):
        raise ValueError('Relative --out paths must stay inside output/; use an absolute path for another location')
    if candidate == OUTPUT_DIR.resolve():
        candidate = OUTPUT_DIR/(source.stem+'.transcript')
    return candidate.with_name(prefix+candidate.name)


def main():
    parser = argparse.ArgumentParser(description='Bounded-memory English transcription with anonymous speakers (AMD ROCm).')
    parser.add_argument('audio', type=Path)
    parser.add_argument('--out', type=Path, help='Run directory under output/; absolute paths override this location')
    parser.add_argument('--no-timestamps', action='store_true', help='Omit time ranges from TXT and terminal output')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('transcribe.toml'))
    parser.add_argument('--check', action='store_true', help='Validate configuration and available resources without decoding or loading models')
    parser.add_argument('--debug', action='store_true', help='Synchronize GPU at diagnostic stage boundaries')
    args = parser.parse_args()
    global DEBUG, MONITOR
    DEBUG = args.debug
    configure_logging(DEBUG)
    try:
        config = Settings.read(args.config)
        source = resolve_audio_path(args.audio)
        requested_out = requested_output_path(args.out, source)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    try:
        out = select_output_directory(requested_out)
        duration = probe_duration(source, config)
        resources = Resources(config, out, torch)
        preflight = resources.preflight(duration)
    except (ValueError, OSError, ResourceLimitError) as error:
        parser.error(str(error))
    MONITOR = resources
    log(f'Preflight passed: {preflight}')
    log('RAM budget is monitored, not an OS reservation; GPU budget caps the PyTorch allocator, not driver allocations.')
    if args.check:
        log('Preflight checks passed.', console=True)
        log(f'Output directory for next run: {out}', console=True)
        log(resources.describe(), console=True)
        return
    out = select_output_directory(requested_out, reserve=True)
    resources.directory = out
    log(f'Output directory: {out}', console=True)
    if out != requested_out:
        log(f'Preserved existing output at {requested_out}; using a fresh directory for this rerun.')
    handler = logging.FileHandler(out/'run.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    LOGGER.addHandler(handler)
    save(out/'effective_config.json', dict(vars(config), include_timestamps=not args.no_timestamps))
    import transformers
    log(f'Python={sys.version.split()[0]}; PyTorch={torch.__version__}; HIP={torch.version.hip}; '
        f'Transformers={transformers.__version__}; GPU={torch.cuda.get_device_properties(0)}')
    log(f'MIOpen immediate inference={config.miopen_immediate}; source={source}; duration={duration:.2f}s')
    log(f'Preflight: {preflight}')
    pcm = out/'decoded.f32'
    store = None
    completed = False
    try:
        with stage('bounded audio decoding'):
            audio = decode_to_disk(source, pcm, config, resources)
        store = Checkpoints(out/'checkpoints.sqlite3', config)
        with stage('streaming speaker detection'):
            diarize(audio, store, config, resources)
        with stage('silence-aware recognition'):
            recognize(audio, store, config, resources)
        export_array(out/'raw_asr.json', store.rows('chunks'), resources)
        with stage('word alignment'):
            align(audio, store, config, resources)
        with stage('speaker attribution and export'):
            export(out, store, source, resources, include_timestamps=not args.no_timestamps,
                   unknown_max_words=config.unknown_max_words,
                   unknown_max_gap_seconds=config.unknown_max_gap_seconds)
        completed = True
        log(f'Saved transcript to {out}', console=True)
        log(resources.describe())
    except torch.OutOfMemoryError as error:
        raise ResourceLimitError(
            f'Inference could not allocate memory with gpu_budget_gib={config.gpu_budget_gib}; '
            f'configured max_chunk_seconds={config.max_chunk_seconds}, max_new_tokens={config.max_new_tokens}. '
            'Check competing GPU usage, lower chunk/token limits, or increase the budget after --check. '
            'Completed work is retained in checkpoints.sqlite3.'
        ) from error
    finally:
        if store is not None:
            try:
                if not (out/'raw_asr.json').exists():
                    export_array(out/'raw_asr.json', store.rows('chunks'), resources)
            except (OSError, RuntimeError) as error:
                LOGGER.warning('Could not export raw_asr.json: %s; SQLite checkpoints are retained', error)
            finally:
                store.close()
        pcm.unlink(missing_ok=True)
        try:
            save(out/'resource_usage.json', dict(completed=completed, **resources.snapshot()))
        except Exception as error:
            LOGGER.warning('Could not save final resource measurements: %s', error)
        handler.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        LOGGER.exception("Transcription failed: %s", error)
        raise SystemExit(1)
