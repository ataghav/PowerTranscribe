"""Validated job configuration, resource preflight, and measured usage (Linux/ROCm)."""
from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tomllib

GIB = 1024 ** 3
MIB = 1024 ** 2


class ResourceLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    ram_budget_gib: float
    gpu_budget_gib: float
    min_free_ram_gib: float
    min_free_gpu_gib: float
    working_disk_gib: float
    min_free_disk_gib: float
    cpu_threads: int
    io_buffer_mib: int
    sqlite_cache_mib: int
    probe_timeout_seconds: int
    max_chunk_seconds: float
    min_chunk_seconds: float
    context_seconds: float
    silence_search_seconds: float
    silence_window_seconds: float
    silence_rms: float
    max_new_tokens: int
    miopen_immediate: bool
    diarization_mode: str
    speaker_threshold: float
    max_segment_seconds: float
    max_segment_gap_seconds: float

    @classmethod
    def read(cls, path):
        with Path(path).open('rb') as file:
            data = tomllib.load(file)
        expected = {f.name for f in fields(cls)}
        missing, unknown = expected - data.keys(), data.keys() - expected
        if missing or unknown:
            raise ValueError(f"Invalid config {path}: missing={sorted(missing)}, unknown={sorted(unknown)}")
        for field in fields(cls):
            value = data[field.name]
            valid = (type(value) is field.type if field.type is not float
                     else type(value) in (float, int) and math.isfinite(value))
            if not valid:
                raise ValueError(f"Config {field.name}: expected {field.type.__name__}, got {value!r}")
            if field.type is int and value > sys.maxsize:
                raise ValueError(f"Config {field.name}: exceeds the platform integer limit")
            zero_allowed = field.name in ('context_seconds', 'silence_rms', 'min_free_ram_gib',
                                           'min_free_gpu_gib', 'min_free_disk_gib', 'max_segment_gap_seconds')
            if field.type in (int, float) and (value < 0 or value == 0 and not zero_allowed):
                condition = 'nonnegative' if zero_allowed else 'positive'
                raise ValueError(f"Config {field.name}: must be {condition}, got {value}")
        settings = cls(**data)
        if not (settings.silence_window_seconds <= settings.silence_search_seconds
                < settings.max_chunk_seconds and settings.min_chunk_seconds <= settings.max_chunk_seconds):
            raise ValueError("Config requires silence_window_seconds <= silence_search_seconds < max_chunk_seconds "
                             "and min_chunk_seconds <= max_chunk_seconds")
        # Qwen's authors limit forced-alignment inputs to 180 seconds.
        if settings.max_chunk_seconds + 2 * settings.context_seconds > 180:
            raise ValueError("Config max_chunk_seconds + 2 * context_seconds exceeds Qwen's 180-second alignment limit")
        if settings.min_chunk_seconds < 0.5:
            raise ValueError("Config min_chunk_seconds must be >= Qwen's 0.5-second minimum input duration")
        if settings.max_segment_seconds > settings.max_chunk_seconds:
            raise ValueError('Config max_segment_seconds must not exceed max_chunk_seconds')
        if settings.probe_timeout_seconds > (2**31-1)//1000:
            raise ValueError('Config probe_timeout_seconds exceeds the subprocess poll timeout range')
        if settings.silence_rms > 1 or settings.speaker_threshold >= 1:
            raise ValueError("Config silence_rms must be <= 1 and speaker_threshold must be < 1")
        if settings.diarization_mode not in ('low_latency', 'very_low_latency', 'ultra_low_latency'):
            raise ValueError("Config diarization_mode must be low_latency, very_low_latency, or ultra_low_latency")
        if settings.cpu_threads > (os.cpu_count() or 1):
            raise ValueError(f"Config cpu_threads={settings.cpu_threads} exceeds {os.cpu_count()} available CPUs")
        tokens = math.ceil((settings.max_chunk_seconds + 2*settings.context_seconds)*13) + settings.max_new_tokens + 128
        if tokens > 65536:
            raise ValueError(f"Config requires approximately {tokens} tokens; Qwen context limit is 65536")
        return settings


def host_memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.strip().split()[0]) * 1024
    total, available = values['MemTotal'], values['MemAvailable']
    # Respect cgroup v2 ceilings as well as the host/WSL VM's physical memory.
    root = Path('/sys/fs/cgroup')
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            current = root / line.split('::', 1)[1].lstrip('/')
            while current == root or root in current.parents:
                maximum, usage = current/'memory.max', current/'memory.current'
                if maximum.exists() and usage.exists():
                    limit = maximum.read_text().strip()
                    if limit != 'max':
                        total = min(total, int(limit))
                        available = min(available, max(0, int(limit)-int(usage.read_text())))
                if current == root:
                    break
                current = current.parent
    return total, available


def process_rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    raise ResourceLimitError('Cannot measure process RAM from /proc/self/status')


def probe_duration(source, config):
    for program in ('ffmpeg', 'ffprobe'):
        if not shutil.which(program):
            raise RuntimeError(f"{program} is missing; install FFmpeg before running transcription")
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
             'stream=duration:format=duration', '-of', 'json', str(source)],
            capture_output=True, text=True, timeout=config.probe_timeout_seconds, check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"ffprobe exceeded probe_timeout_seconds={config.probe_timeout_seconds}") from error
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Cannot inspect audio: {error.stderr.strip()}") from error
    info = json.loads(result.stdout)
    if not info.get('streams'):
        raise ValueError('Input contains no audio stream')
    value = info['streams'][0].get('duration', info.get('format', {}).get('duration'))
    try:
        duration = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('Input has no reliable duration; cannot validate working-disk requirements') from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid audio duration: {value!r}")
    return duration


class Resources:
    def __init__(self, config, directory, torch_module):
        self.config, self.directory, self.torch = config, Path(directory), torch_module
        self.peak_rss = 0

    def preflight(self, duration):
        cfg, torch = self.config, self.torch
        if not torch.version.hip or not torch.cuda.is_available():
            raise ResourceLimitError('ROCm GPU unavailable; check the AMD driver and gfx1151 PyTorch installation')
        _, available = host_memory()
        free, total = torch.cuda.mem_get_info()
        parent = self.directory
        while not parent.exists():
            parent = parent.parent
        disk_free = shutil.disk_usage(parent).free
        # Envelope for the pinned FP16 Qwen weights, KV cache, and working buffers.
        # This is a conservative startup estimate, not a hardware reservation.
        input_tokens = math.ceil((cfg.max_chunk_seconds + 2*cfg.context_seconds)*13) + 128
        kv_gib = 2*28*8*128*2*(input_tokens + cfg.max_new_tokens)/GIB
        required_gpu = 5.0 + kv_gib
        required_ram = 7.0 + (cfg.io_buffer_mib*8 + cfg.sqlite_cache_mib)/1024
        if cfg.gpu_budget_gib < required_gpu or cfg.ram_budget_gib < required_ram:
            raise ResourceLimitError(f"Configured budgets are too small for the pinned models/input limits: "
                                     f"need an estimated GPU >= {required_gpu:.2f} GiB and RAM >= {required_ram:.2f} GiB; "
                                     f"configured GPU={cfg.gpu_budget_gib}, RAM={cfg.ram_budget_gib} GiB")
        for name, budget, headroom, actual in (
            ('RAM', cfg.ram_budget_gib, cfg.min_free_ram_gib, available/GIB),
            ('GPU', cfg.gpu_budget_gib, cfg.min_free_gpu_gib, free/GIB),
            ('working disk', cfg.working_disk_gib, cfg.min_free_disk_gib, disk_free/GIB),
        ):
            if budget + headroom > actual:
                raise ResourceLimitError(f"Insufficient {name}: budget {budget:.2f} + headroom {headroom:.2f} GiB "
                                         f"requires {budget+headroom:.2f} GiB; only {actual:.2f} GiB available")
        pcm_bytes = math.ceil((duration+1)*16000)*4
        # Include conservative metadata/export space; actual disk use is checked throughout.
        estimated_disk = pcm_bytes + math.ceil(duration)*32768
        if estimated_disk > cfg.working_disk_gib*GIB:
            raise ResourceLimitError(f"Recording needs an estimated {estimated_disk/GIB:.2f} GiB working disk; "
                                     f"working_disk_gib={cfg.working_disk_gib}. Increase it or split the recording")
        torch.set_num_threads(cfg.cpu_threads)
        torch.cuda.set_per_process_memory_fraction(cfg.gpu_budget_gib*GIB/total)
        torch.cuda.reset_peak_memory_stats()
        self.check()
        return dict(estimated_gpu_gib=required_gpu, estimated_ram_gib=required_ram,
                    estimated_working_disk_gib=estimated_disk/GIB,
                    available_ram_gib=available/GIB, available_gpu_gib=free/GIB, available_disk_gib=disk_free/GIB)

    def disk_bytes(self):
        return sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file()) if self.directory.exists() else 0

    def check_disk(self, additional=0):
        used = self.disk_bytes()
        existing = self.directory
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free
        if used + additional > self.config.working_disk_gib*GIB:
            raise ResourceLimitError(f"Working disk budget exceeded: {(used+additional)/GIB:.2f} GiB > "
                                     f"working_disk_gib={self.config.working_disk_gib}; checkpoints are retained")
        if free - additional < self.config.min_free_disk_gib*GIB:
            raise ResourceLimitError(f"Disk headroom would fall below min_free_disk_gib={self.config.min_free_disk_gib}")

    def check(self):
        rss = process_rss()
        self.peak_rss = max(self.peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        _, available = host_memory()
        if self.peak_rss > self.config.ram_budget_gib*GIB:
            raise ResourceLimitError(f"Process RAM peak {self.peak_rss/GIB:.2f} GiB exceeded "
                                     f"ram_budget_gib={self.config.ram_budget_gib}")
        if available < self.config.min_free_ram_gib*GIB:
            raise ResourceLimitError(f"Available RAM {available/GIB:.2f} GiB is below "
                                     f"min_free_ram_gib={self.config.min_free_ram_gib}")
        self.check_disk()
        return rss

    def snapshot(self):
        torch = self.torch
        rss = process_rss()
        self.peak_rss = max(self.peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        free, total = torch.cuda.mem_get_info()
        host_total, host_available = host_memory()
        return dict(process_ram_gib=rss/GIB, process_peak_ram_gib=self.peak_rss/GIB,
                    child_peak_ram_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024/GIB,
                    host_total_ram_gib=host_total/GIB, host_available_ram_gib=host_available/GIB,
                    gpu_allocated_gib=torch.cuda.memory_allocated()/GIB,
                    gpu_reserved_gib=torch.cuda.memory_reserved()/GIB,
                    gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,
                    gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,
                    gpu_free_gib=free/GIB, gpu_total_gib=total/GIB,
                    working_disk_gib=self.disk_bytes()/GIB)

    def describe(self):
        values = self.snapshot()
        return (f"RAM GiB: process={values['process_ram_gib']:.2f}, peak={values['process_peak_ram_gib']:.2f}, "
                f"host_available={values['host_available_ram_gib']:.2f}; GPU GiB: "
                f"allocated={values['gpu_allocated_gib']:.2f}, reserved={values['gpu_reserved_gib']:.2f}, "
                f"peak={values['gpu_peak_allocated_gib']:.2f}, free={values['gpu_free_gib']:.2f}")
