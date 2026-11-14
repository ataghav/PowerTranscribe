# Meeting transcriber

Local English transcription on Linux / WSL2 / AMD Radeon 8060S (`gfx1151`). Qwen3-ASR supplies words, Qwen3 ForcedAligner supplies timestamps, and Nemotron 3 supplies anonymous speaker labels. Models run sequentially on the GPU; there is no CPU inference fallback.

## Install and run

Install `uv`, `ffmpeg` (including `ffprobe`), and a compatible AMD ROCm host driver, then:

```bash
uv sync
uv run python transcribe.py meeting.m4a
```

Put recordings in `input/`. A bare filename is resolved there; explicit audio paths are also accepted. Results default to `output/<recording>.transcript/`. Relative `--out` names are placed under `output/`; absolute paths explicitly select another location. The input and output folders are ignored by Git except for their `.gitkeep` placeholders.

Models are managed separately from Git. Each run first checks the installed snapshots under `models/`; complete snapshots load locally without contacting Hugging Face. Missing snapshots download at the exact revisions in [models.json](models.json), using the Hugging Face cache where available, and are copied into self-contained local directories. Existing incomplete snapshots fail explicitly rather than being silently replaced. Model storage and the Hugging Face cache need additional disk space beyond the per-job working directory.

## Model backup and restore

I store a backup of the model weights in my personal OneDrive account. The backup is managed manually; these scripts do not sign in to or upload to OneDrive. Neither the weights nor ZIP archives are committed to Git.

From the repository root, download all three pinned model snapshots and create the backup:

```bash
uv run python docs/download_models.py
```

This installs the models under `models/` and writes `docs/powertranscribe-models.zip`. Copy that ZIP to your personal OneDrive. Existing archives are never overwritten; use `--archive /path/to/new-backup.zip` for a different destination. The ZIP uses ZIP64 and no compression to avoid spending CPU compressing large weight files. Allow room for the Hugging Face cache, installed snapshots, and ZIP simultaneously. An interrupted archive creation can leave an incomplete ZIP; remove that incomplete file or choose a new destination before retrying.

The archive includes complete model snapshots (weights, configuration, tokenizer/processor files, and repository documentation/license files) and a per-model manifest recording repository, pinned revision, file sizes, and SHA-256 checksums. Cache symlinks are copied as real files. Existing snapshots are fully checksum-verified before archiving.

To restore on another machine, first make the ZIP available locally from OneDrive, then run:

```bash
python3 docs/restore_models.py /path/to/powertranscribe-models.zip
```

Restore uses only the Python standard library, extracts into a temporary directory, and verifies all models against the revisions expected by this checkout before installation. It rejects checksum failures and unsafe archive paths. Existing valid installations are verified and retained; existing damaged installations are reported and must be moved aside before retrying. Use only your trusted backup: checksums detect corruption, not a maliciously replaced archive and manifest. Retain the models' license files and terms with the backup.

Both scripts and transcription respect `POWERTRANSCRIBE_MODELS=/absolute/path/to/models` to select another installation directory. Normal runs check manifest identities and file sizes without rehashing gigabytes of weights; backup and restore perform full checksum verification.

After installing Python dependencies and restoring models, test without Hub access:

```bash
HF_HUB_OFFLINE=1 uv run --offline python transcribe.py meeting.m4a
```

Missing models then produce an error rather than a network download. The archive contains model artifacts only: Python packages, FFmpeg, ROCm, and drivers still need separate installation. `uv.lock` pins the Python environment but does not back up the package downloads.

## Run options

Validate configuration, audio metadata, and available resources before decoding or loading model weights:

```bash
uv run python transcribe.py meeting.m4a --check
```

Choose a configuration and a run name:

```bash
uv run python transcribe.py meeting.m4a --config transcribe.toml --out result.transcript --debug

# Omit leading time ranges from the TXT file and terminal output:
uv run python transcribe.py meeting.m4a --out plain.transcript --no-timestamps
```

Existing output directories are preserved. Repeating the same command automatically selects a fresh numbered sibling, such as `result-2.transcript/`, and prints the selected directory. `--debug` enables detailed console diagnostics, model-loading progress bars, tracebacks, and GPU synchronization at stage boundaries. Without it, the console shows the output location, transcript, completion message, and actionable warnings/errors. Detailed stage and memory logs remain in `run.log`; full failure tracebacks are saved there even in normal mode. `--check` explicitly prints its resource-validation summary. Use `--help` for the CLI options.

## Limits and resource validation

All operational budgets and chunking settings are in [transcribe.toml](transcribe.toml). The file is required, and unknown keys, missing keys, wrong types, nonfinite numbers, and inconsistent bounds are rejected. The default file sits beside the script, independent of the current working directory.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `ram_budget_gib` | 12 | Monitored process resident-memory ceiling, including model loading |
| `gpu_budget_gib` | 8 | PyTorch GPU allocator ceiling |
| `min_free_ram_gib` / `min_free_gpu_gib` | 2 / 1 | Startup headroom; host RAM headroom is also checked during processing |
| `working_disk_gib` | 16 | Decoded PCM, SQLite checkpoints, logs, and exports in the output directory |
| `min_free_disk_gib` | 2 | Disk headroom checked before writing and between processing steps |
| `cpu_threads` | 4 | PyTorch and FFmpeg worker-thread setting |
| `io_buffer_mib` / `sqlite_cache_mib` | 1 / 8 | Bounded decoder I/O and SQLite page cache |
| `max_chunk_seconds` | 40 | Hard maximum ASR core duration; context is added separately |
| `context_seconds` | 1 | ASR/alignment context on each side; maximum model input is therefore 42 seconds by default |
| `silence_search_seconds` / `silence_window_seconds` | 5 / 0.3 | Search backward near the core limit for the last quiet window |
| `silence_rms` | 0.001 | Quiet-window RMS threshold; without silence, use the lowest-energy window |
| `min_chunk_seconds` / `max_new_tokens` | 10 / 2048 | Stop retrying token-limited cores at this duration; never accept truncated output |
| `diarization_mode` | `low_latency` | NVIDIA's supported streaming preset; alternatives are validated by name |
| `miopen_immediate` | `true` | Native PyTorch/MIOpen convolution selection mode during inference |
| `unknown_max_words` | 2 | Maximum consecutive UNKNOWN words eligible for contextual speaker inference; 0 disables |
| `unknown_max_gap_seconds` | 0.5 | Maximum gap between each neighboring word in a contextual fill, including both bounding words; 0 disables |

Startup checks compare budgets plus headroom against host/WSL RAM (including cgroup v2 limits when present), GPU free memory, and free working-disk space. They also check model input/context limits, a conservative model-memory envelope, CPU count, and estimated disk requirements from the recording duration. Error messages name the offending setting and the required/available capacity. Unknown-duration or invalid audio is rejected because its disk requirement cannot be checked reliably.

These checks **do not reserve memory or disk space** against other applications. The RAM ceiling is checked at processing boundaries using current RSS and the kernel-recorded process peak; it is not an OS-enforced memory limit. PyTorch's allocator limit does not cover every driver/native allocation. For strict host isolation, deploy inside a cgroup/container with an appropriate memory limit in addition to these checks. Configuration cannot promise that arbitrary inputs, changing system load, driver failures, or exhausted storage will never cause failure. Resource failures are reported explicitly, and committed SQLite work is retained.

## Bounded processing of long recordings

FFmpeg's float32 mono 16 kHz output is copied to disk through bounded buffers. Inference reads bounded spans from that file; it never loads the complete recording into RAM or builds complete-recording GPU features.

Nemotron uses its documented streaming processor and carries its native speaker cache from one call to the next. Speaker IDs therefore do not restart at recognition boundaries. Completed speaker turns are written to SQLite while at most eight open turns remain in memory. Streaming diarization can produce different boundaries/labels than the earlier offline whole-recording mode.

ASR searches **before** the configured core limit for the last quiet window. If none exists, it cuts in the lowest-energy window within the search interval. All source samples belong to exactly one core; overlapping context protects boundary words, and alignment keeps only words whose midpoint belongs to that core. There is no file-size threshold or fixed maximum recording duration: working disk and processing time grow with duration, while audio/model buffers and intermediate-result memory stay bounded. Larger jobs may require increasing the disk budget.

If the model nevertheless reaches the generation limit, the core is split and retried within configured bounds. No token limit is increased automatically, and no truncated output is silently accepted. ASR chunks, speaker turns, words, and segments live in SQLite; final JSON/text/SRT exports are streamed from it rather than assembled in recording-sized Python lists.

## Outputs and memory measurements

Consecutive words with the same speaker label are combined until that label changes, including across pauses and recognition chunks. The merged range starts at the first word and ends at the latest word end. The old `max_segment_seconds` and `max_segment_gap_seconds` settings have been removed. Segment text is streamed from its member words in SQLite, so an uninterrupted speaker does not require a recording-sized string in RAM.

Before grouping, short UNKNOWN runs can inherit a speaker from their immediate neighbors. By default, at most two UNKNOWN words are filled only when both bounding words have the same unambiguous speaker label and every adjacent gap is between 0 and 0.5 seconds. Runs at recording edges, longer runs, overlapping word intervals, differing bounding speakers, overlap labels, or conflicting detected speaker candidates remain unchanged. This uses bounded look-ahead, including across recognition chunks. Set either `unknown_max_words` or `unknown_max_gap_seconds` to 0 to disable it. Filled JSON words retain `original_speaker: "UNKNOWN"` and `speaker_inference: "matching_neighbors"`; inferred labels are contextual estimates, not new detector evidence. TXT/SRT and JSON segments use the filled labels and merge accordingly; word timestamps are unchanged. Existing exports are not modified automatically.

`--no-timestamps` controls leading ranges in TXT/terminal output. JSON retains word/segment timing, and SRT retains the timing required for subtitles. SRT cues follow the same speaker-turn grouping, so a long uninterrupted speaker produces a long cue.

The output directory contains:

- `transcript.txt`, `transcript.srt`, `transcript.json` — final timestamped transcript and speaker labels.
- `raw_asr.json` — streamed recognition export, also attempted on failure.
- `checkpoints.sqlite3` — durable completed intermediate results. Automatic resume is not implemented.
- `effective_config.json` — settings and timestamp-display option used for this run.
- `run.log` — progress, stage timings, process RAM/GPU snapshots, and failure traceback.
- `resource_usage.json` — final current/peak process RAM, child-process RAM peak, GPU allocated/reserved peaks, host/GPU availability, and a `completed` flag.

Decoded PCM is removed on normal exit or handled failure. After forced termination, it may remain for manual cleanup. Final transcript files are first written as `.part` files and renamed after export succeeds.

RAM measurements are Linux process RSS/high-water values; child-process peaks may include pages inherited across fork. GPU *allocated* is live PyTorch tensor memory, while *reserved* includes its allocator cache: these overlap and must not be added together. Driver free memory is device-wide. The Radeon APU shares physical memory with the host, so host and GPU figures are not independent totals.

Measured on a 90-second excerpt containing the previously failing interval: direct-to-GPU model loading reduced peak process RAM from **10.35 to 6.36 GiB**; peak GPU tensor allocation was **4.18 GiB**, with **4.55 GiB** peak allocator reservation. These are measured examples, not universal per-file requirements. Streaming diarization used approximately **0.46 GiB** peak GPU tensor allocation.

The full 76.4-minute panel recording completed with 122 continuous, silence-aware cores, no generation-limit retry, and 5,909 aligned words (676 segments under the earlier time-limited formatting). Peak process RAM was **6.34 GiB**; peak GPU allocation/reservation was **4.18 / 4.81 GiB**. Native low-latency streaming trades throughput for bounded state: speaker detection took about 293 seconds, compared with about 11 seconds for the earlier whole-recording GPU forward. Recognition took about 288 seconds. Transcript accuracy was not manually scored.

## Diagnosis and validation

Native MIOpen output bypasses Python logging. To capture it:

```bash
set -o pipefail
MIOPEN_ENABLE_LOGGING=1 MIOPEN_ENABLE_LOGGING_CMD=1 MIOPEN_LOG_LEVEL=6 \
  uv run python transcribe.py meeting.m4a --out miopen-debug.transcript --debug 2>&1 | tee miopen-debug.log
```

Detailed logs can be large. Keep the working-disk budget in mind.

Run the CPU-side configuration, boundary, checkpoint, and export checks:

```bash
uv run python -m unittest discover -s tests -v
```

Speaker labels are anonymous and recording-local, with at most eight speakers per recording. Overlap and uncertainty are explicit. Diarization and ASR accuracy are model properties; successful execution does not establish transcript accuracy.

## Authoritative implementation references

- [Qwen's author-maintained chunker and input limits](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py): low-energy boundaries, 1,200-second ASR inputs, 180-second alignment inputs, and a 0.5-second ASR minimum. This application caps the combined ASR/alignment input at 180 seconds and searches only backward so a boundary cannot exceed the configured limit.
- [NVIDIA's streaming diarization example](https://huggingface.co/nvidia/Nemotron-3-Diarization#streaming-diarization): native processor window sizes and speaker-cache continuation. This implementation consumes each chunk's activity immediately instead of retaining/concatenating all logits.
- [Hugging Face model loading](https://huggingface.co/docs/transformers/main/models): device-map loading without constructing a full CPU model and then copying it to the GPU. The explicit device map here places every layer on the AMD GPU.
- [AMD MIOpen immediate mode](https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html) and [PyTorch's supported flag](https://docs.pytorch.org/docs/2.14/backends.html#torch.backends.miopen.immediate): bypassing the failing solver-timing search through a native backend option, not replacing model layers. This mitigates the observed backend failure; it does not repair the underlying driver/library timing issue.
- [PyTorch allocator budget](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.set_per_process_memory_fraction.html), [ROCm memory behavior](https://docs.pytorch.org/docs/main/notes/hip.html), and [Linux cgroup memory limits](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html): the distinction between application checks, allocator ceilings, and OS-enforced resource isolation.
