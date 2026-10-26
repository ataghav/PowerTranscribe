#!/usr/bin/env python3
"""uv run python transcribe.py meeting.wav [--out result]"""
import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
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
REVISIONS = {
    ASR: "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
    ALIGNER: "c07281df297b9905d24a508279258cccf987a064",
    DIARIZER: "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
}


def log(message):
    print(message, file=sys.stderr, flush=True)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def decode(path):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is missing. Install it with: sudo apt install ffmpeg")
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or invalid.")
    return audio


def load(repo, model_class, dtype=torch.float16):
    processor = AutoProcessor.from_pretrained(repo, revision=REVISIONS[repo])
    model = model_class.from_pretrained(
        repo, revision=REVISIONS[repo], dtype=dtype,
    ).to(DEVICE).eval()
    return processor, model


def release():
    gc.collect()
    torch.cuda.empty_cache()


def diarize(audio):
    log("1/3 Detecting speakers across the complete recording...")
    processor, model = load(DIARIZER, AutoModelForAudioFrameClassification, torch.float32)
    inputs = processor(audio, sampling_rate=SR, return_tensors="pt").to(DEVICE, torch.float32)
    with torch.inference_mode():
        output = model(**inputs)
    segments = processor.extract_speaker_dict(output.logits, inputs.attention_mask, threshold=.5)[0]
    duration = len(audio) / SR
    turns = [dict(start=max(0., float(s["Start"])), end=min(duration, float(s["End"])),
                  speaker=f'SPEAKER_{int(s["Speaker"]) + 1:02d}') for s in segments]
    turns = sorted((t for t in turns if t["end"] > t["start"]), key=lambda t: t["start"])
    del model, processor, inputs, output
    release()
    return turns


def recognize(audio, out):
    processor, model = load(ASR, AutoModelForMultimodalLM)
    duration = len(audio) / SR
    chunks = []
    count = math.ceil(duration / 40)
    for i in range(count):
        core_start, core_end = i * 40., min(duration, (i + 1) * 40.)
        start, end = max(0., core_start - 1.), min(duration, core_end + 1.)
        clip = audio[round(start * SR):round(end * SR)]
        log(f"2/3 Recognizing words: {i + 1}/{count}")
        text = ""
        if np.max(np.abs(clip)) > 1e-5:
            inputs = processor.apply_transcription_request(audio=clip, language="English").to(DEVICE, torch.float16)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=2048, do_sample=False, num_beams=1)
            ids = output[:, inputs["input_ids"].shape[1]:]
            if ids.shape[1] >= 2048:
                raise RuntimeError("Recognizer reached its token limit; refusing a truncated transcript.")
            text = processor.decode(ids, return_format="transcription_only")[0].strip()
            del inputs, output, ids
        chunks.append(dict(start=start, end=end, core_start=core_start, core_end=core_end, text=text))
        save(out / "raw_asr.json", chunks)
    del model, processor
    release()
    return chunks


def align(audio, chunks):
    processor, model = load(ALIGNER, AutoModelForTokenClassification)
    words = []
    for i, chunk in enumerate(chunks):
        log(f"3/3 Aligning words: {i + 1}/{len(chunks)}")
        if not chunk["text"]:
            continue
        clip = audio[round(chunk["start"] * SR):round(chunk["end"] * SR)]
        inputs, word_lists = processor.prepare_forced_aligner_inputs(
            audio=clip, transcript=chunk["text"], language="English",
        )
        inputs = inputs.to(DEVICE, torch.float16)
        with torch.inference_mode():
            output = model(**inputs)
        aligned = processor.decode_forced_alignment(
            logits=output.logits, input_ids=inputs["input_ids"], word_lists=word_lists,
            timestamp_token_id=model.config.timestamp_token_id,
        )[0]
        if not aligned:
            raise RuntimeError("Alignment returned no words. Recognition text is retained in raw_asr.json.")
        for word in aligned:
            a, b = float(word["start_time"]), float(word["end_time"])
            if not math.isfinite(a + b) or b < a:
                raise ValueError("Invalid alignment timestamp.")
            a = min(len(clip) / SR, max(0., a)) + chunk["start"]
            b = min(len(clip) / SR, max(0., b)) + chunk["start"]
            midpoint = (a + b) / 2
            if chunk["core_start"] <= midpoint < chunk["core_end"]:
                words.append(dict(start=a, end=b, text=word["text"]))
        del inputs, output
    del processor, model
    release()
    return sorted(words, key=lambda w: (w["start"], w["end"]))


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


def export(out, words, turns, source):
    segments = []
    for word in words:
        if (segments and segments[-1]["speaker"] == word["speaker"]
                and word["start"] - segments[-1]["end"] < .8
                and word["end"] - segments[-1]["start"] <= 12):
            segments[-1]["text"] += " " + word["text"]
            segments[-1]["end"] = max(segments[-1]["end"], word["end"])
        else:
            segments.append({key: word[key] for key in ("start", "end", "text", "speaker")})
    text, subtitles = [], []
    for i, segment in enumerate(segments, 1):
        content = f'{segment["speaker"]}: {segment["text"]}'
        text.append(f'[{timestamp(segment["start"])}–{timestamp(segment["end"])}] {content}')
        subtitles.append(f'{i}\n{timestamp(segment["start"], True)} --> '
                         f'{timestamp(max(segment["end"], segment["start"] + .05), True)}\n{content}\n')
    transcript = "\n".join(text) + "\n"
    (out / "transcript.txt").write_text(transcript, encoding="utf-8")
    (out / "transcript.srt").write_text("\n".join(subtitles), encoding="utf-8")
    save(out / "transcript.json", dict(source=str(source), models=REVISIONS,
                                      words=words, segments=segments, diarization=turns))
    print(transcript, end="")


def main():
    parser = argparse.ArgumentParser(description="Local English transcription with anonymous speaker labels (AMD ROCm).")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, help="Output directory; defaults to <audio>.transcript")
    args = parser.parse_args()
    source = args.audio.expanduser().resolve(strict=True)
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU unavailable. Check the AMD driver and gfx1151 PyTorch installation.")
    out = args.out or source.with_suffix(".transcript")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Output is not empty: {out}. Choose a fresh --out directory.")
    out.mkdir(parents=True, exist_ok=True)
    audio = decode(source)
    log(f"Loaded {len(audio) / SR / 60:.1f} minutes. GPU: {torch.cuda.get_device_name(0)}")
    turns = diarize(audio)
    chunks = recognize(audio, out)
    words = align(audio, chunks)
    attribute(words, turns)
    export(out, words, turns, source)
    log(f"Saved transcript to {out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        log(f"Error: {error}")
        raise SystemExit(1)
