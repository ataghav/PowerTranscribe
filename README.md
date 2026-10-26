# Meeting transcriber

A deliberately small terminal prototype for **English audio on Linux / Ryzen AI Max+ 395 / Radeon 8060S**. It uses Qwen3-ASR 1.7B for words, Qwen3 ForcedAligner for timestamps, and Nemotron 3 for anonymous speaker labels. No enrollment, voice profiles, cloud transcription, or model-selection menu.

## Install

Prerequisites: `uv`, `ffmpeg`, and a working ROCm 10.1-compatible host kernel/driver. This project installs AMD's gfx1151 PyTorch wheels; it does not install your system driver.

```bash
cd meeting-transcriber
uv sync
```

`uv` selects Python 3.12, creates the project-local `.venv`, and writes `uv.lock`. Keep the generated lockfile. No manual activation is needed. On Ubuntu, install FFmpeg with `sudo apt install ffmpeg` if missing. Install uv using its [official instructions](https://docs.astral.sh/uv/getting-started/installation/).

## Run

```bash
uv run python transcribe.py /path/to/meeting.mp3
```

Or choose an output directory:

```bash
uv run python transcribe.py meeting.wav --out result
```

The first run downloads model weights. Subsequent runs reuse the cache. The script prints progress and the final transcript, and writes:

- `transcript.txt` — readable timestamped speaker turns.
- `transcript.srt` — speaker-labelled subtitles.
- `transcript.json` — word timestamps, speaker turns and overlap flags.
- `raw_asr.json` — recognition text saved before alignment.

By default output goes next to the recording in `<filename>.transcript/`. Use a new output directory for a rerun; existing nonempty directories are protected.

Illustrative output:

```text
[00:00:01.200–00:00:03.800] SPEAKER_01: Let's start with the deployment.
[00:00:04.000–00:00:05.700] SPEAKER_02: The tests are finished.
[00:00:06.100–00:00:07.500] SPEAKER_01: Great, let's proceed.
```

## Fixed choices

- Diarization runs across the whole file, preserving the model's speaker state. Speaker IDs do not restart at each recognition chunk.
- Recognition uses 40-second cores with one second of context; alignment restores original timestamps before assigning speakers.
- Up to **eight distinct speakers per recording**. Labels are local to that recording; `SPEAKER_01` in another file can be a different person.
- Overlapping speech and uncertain boundaries receive explicit labels. The prototype does not recover two complete word streams from simultaneous speech.
- The input is downmixed to mono. This is for a mixed meeting recording, not separately recorded participant channels.
- Models run sequentially on the AMD GPU. Enrollment and pyannote dependencies have been removed.

Diarization can still merge or split people; consistent labels are the model's objective, not a guarantee. No tests or hardware inference were run for this simplified prototype.

## Setup references

- [AMD compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html): select the Ryzen APU / gfx1151 row and match the host OS/kernel.
- [AMD PyTorch wheels](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
- [uv projects and local virtual environments](https://docs.astral.sh/uv/guides/projects/).
- [Qwen ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf), [aligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf), [Nemotron diarizer](https://huggingface.co/nvidia/Nemotron-3-Diarization).

The dependencies and model revisions were selected from the October 26, 2026 research. This is a Linux/AMD prototype; the earlier Mac and alternative-model paths are intentionally outside this smaller project.
