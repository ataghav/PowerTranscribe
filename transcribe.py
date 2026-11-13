#!/usr/bin/env python3
"""uv run python transcribe.py meeting.wav [--out result]"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
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
DEBUG = False
MONITOR = None
REVISIONS = {
    ASR: "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
    ALIGNER: "c07281df297b9905d24a508279258cccf987a064",
    DIARIZER: "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
}


def log(message):
    LOGGER.info(message)


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
        LOGGER.error("Failed %s after %.1fs; %s", name, time.monotonic() - started, gpu_memory())
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
        processor = AutoProcessor.from_pretrained(repo, revision=REVISIONS[repo])
        model = model_class.from_pretrained(
            repo, revision=REVISIONS[repo], dtype=dtype, device_map={"": DEVICE},
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


def export(out, store, source, config, resources):
    segment = None
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
        if (segment and segment['speaker'] == word['speaker']
                and word['start']-segment['end'] < config.max_segment_gap_seconds
                and word['end']-segment['start'] <= config.max_segment_seconds):
            segment['text'] += ' '+word['text']
            segment['end'] = max(segment['end'], word['end'])
        else:
            if segment:
                store.add('segments', [segment])
            segment = {key: word[key] for key in ('start', 'end', 'text', 'speaker')}
        store.update_word(identity, word)
    if segment:
        store.add('segments', [segment])
    with (out/'transcript.txt.part').open('w', encoding='utf-8') as text_file, \
            (out/'transcript.srt.part').open('w', encoding='utf-8') as srt_file:
        for i, segment in enumerate(store.rows('segments'), 1):
            content = f'{segment["speaker"]}: {segment["text"]}'
            line = f'[{timestamp(segment["start"])}–{timestamp(segment["end"])}] {content}\n'
            subtitle = (f'{i}\n{timestamp(segment["start"], True)} --> '
                        f'{timestamp(max(segment["end"], segment["start"]+.05), True)}\n{content}\n\n')
            resources.check_disk(len(line.encode('utf-8'))+len(subtitle.encode('utf-8')))
            text_file.write(line)
            srt_file.write(subtitle)
            print(line, end='')
    with (out/'transcript.json.part').open('w', encoding='utf-8') as file:
        file.write('{"source":'+json.dumps(str(source))+',"models":'+json.dumps(REVISIONS))
        for key, table in [('words', 'words'), ('segments', 'segments'), ('diarization', 'turns')]:
            file.write(','+json.dumps(key)+':')
            write_json_array(file, store.rows(table), resources)
        file.write('}')
    for name in ('transcript.txt', 'transcript.srt', 'transcript.json'):
        (out/(name+'.part')).replace(out/name)


def main():
    parser = argparse.ArgumentParser(description='Bounded-memory English transcription with anonymous speakers (AMD ROCm).')
    parser.add_argument('audio', type=Path)
    parser.add_argument('--out', type=Path, help='Output directory; defaults to <audio>.transcript')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('transcribe.toml'))
    parser.add_argument('--check', action='store_true', help='Validate configuration and available resources without decoding or loading models')
    parser.add_argument('--debug', action='store_true', help='Synchronize GPU at diagnostic stage boundaries')
    args = parser.parse_args()
    global DEBUG, MONITOR
    DEBUG = args.debug
    try:
        config = Settings.read(args.config)
        source = args.audio.expanduser().resolve(strict=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    out = (args.out or source.with_suffix('.transcript')).expanduser().resolve()
    if out.exists() and (not out.is_dir() or not args.check and any(out.iterdir())):
        parser.error(f'Output is not empty or not a directory: {out}. Choose a fresh --out directory.')
    try:
        duration = probe_duration(source, config)
        resources = Resources(config, out, torch)
        preflight = resources.preflight(duration)
    except (ValueError, OSError, ResourceLimitError) as error:
        parser.error(str(error))
    MONITOR = resources
    log(f'Preflight passed: {preflight}')
    log('RAM budget is monitored, not an OS reservation; GPU budget caps the PyTorch allocator, not driver allocations.')
    if args.check:
        log(resources.describe())
        return
    out.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out/'run.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    LOGGER.addHandler(handler)
    save(out/'effective_config.json', vars(config))
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
            export(out, store, source, config, resources)
        completed = True
        log(f'Saved transcript to {out}; {resources.describe()}')
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
    LOGGER.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(console)
    LOGGER.propagate = False
    try:
        main()
    except Exception as error:
        LOGGER.exception("Transcription failed: %s", error)
        raise SystemExit(1)
