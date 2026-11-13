"""Bounded PCM reads, silence-aware boundaries, and durable on-disk intermediates."""
from collections import deque
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import threading

import numpy as np

SR = 16000


class AudioFile:
    def __init__(self, path, max_read_seconds):
        self.path = Path(path)
        size = self.path.stat().st_size
        if not size or size % 4:
            raise ValueError('Decoded audio is empty or has an incomplete float32 sample')
        self.samples = size // 4
        self.duration = self.samples / SR
        self.max_read_samples = math.ceil(max_read_seconds*SR)

    def read_samples(self, start, count):
        if start < 0 or count < 0 or count > self.max_read_samples or start+count > self.samples:
            raise ValueError(f"Invalid/beyond-budget PCM read: start={start}, count={count}, "
                             f"buffer_limit={self.max_read_samples} samples")
        with self.path.open('rb') as file:
            file.seek(start*4)
            audio = np.fromfile(file, dtype='<f4', count=count)
        if len(audio) != count:
            raise OSError('Decoded audio was truncated while processing')
        return audio

    def read(self, start, end):
        a, b = round(start*SR), round(end*SR)
        return self.read_samples(a, b-a)


def decode_to_disk(source, target, config, resources):
    """No unbounded subprocess stdout/stderr capture, including on malformed input."""
    block_size = config.io_buffer_mib*1024**2
    errors = deque(maxlen=8)
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', str(config.cpu_threads),
               '-i', str(source), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(SR),
               '-f', 'f32le', 'pipe:1']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def drain_errors():
        while block := process.stderr.read1(block_size):
            errors.append(block)

    reader = threading.Thread(target=drain_errors, daemon=True)
    reader.start()
    try:
        with Path(target).open('wb') as file:
            while block := process.stdout.read(block_size):
                resources.check()
                resources.check_disk(len(block))
                if len(block) % 4 or not np.isfinite(np.frombuffer(block, dtype='<f4')).all():
                    raise ValueError('Decoder returned invalid float32 audio')
                file.write(block)
        returncode = process.wait()
        reader.join()
        if returncode:
            message = b''.join(errors).decode(errors='replace').strip()
            raise ValueError(f"FFmpeg failed ({returncode}): {message}")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join(timeout=1)
        process.stdout.close()
        process.stderr.close()
    return AudioFile(target, config.max_chunk_seconds + 2*config.context_seconds)


def silence_boundary(audio, start, limit, config):
    """Last quiet window before the hard boundary; never extends the configured limit."""
    left = max(start+config.min_chunk_seconds, limit-config.silence_search_seconds)
    if limit-left < config.silence_window_seconds:
        return limit
    clip = audio.read(left, limit)
    window = max(1, round(config.silence_window_seconds*SR))
    # Non-overlapping windows bound both computation and temporary memory.
    count = len(clip)//window
    if not count:
        return limit
    energy = np.mean(clip[:count*window].reshape(count, window).astype(np.float64)**2, axis=1)
    quiet = np.flatnonzero(energy <= config.silence_rms**2)
    index = int(quiet[-1]) if len(quiet) else int(np.argmin(energy))
    boundary = left + (index+0.5)*window/SR
    return min(limit, max(start+config.min_chunk_seconds, round(boundary*SR)/SR))


class Checkpoints:
    def __init__(self, path, config):
        self.connection = sqlite3.connect(path)
        self.connection.execute('PRAGMA journal_mode=DELETE')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('PRAGMA temp_store=FILE')
        self.connection.execute(f'PRAGMA cache_size={-config.sqlite_cache_mib*1024}')
        self.connection.executescript('''
            CREATE TABLE chunks (id INTEGER PRIMARY KEY, start REAL, end REAL, data TEXT NOT NULL);
            CREATE TABLE turns (id INTEGER PRIMARY KEY, start REAL, end REAL, data TEXT NOT NULL);
            CREATE INDEX turns_order ON turns(start, end, id);
            CREATE TABLE words (id INTEGER PRIMARY KEY, start REAL, end REAL, data TEXT NOT NULL, segment_id INTEGER);
            CREATE INDEX words_order ON words(start, end, id);
            CREATE INDEX words_segment ON words(segment_id, start, end, id);
            CREATE TABLE segments (id INTEGER PRIMARY KEY, start REAL, end REAL, data TEXT NOT NULL);
        ''')

    def add(self, table, values):
        if table not in ('chunks', 'turns', 'words', 'segments'):
            raise ValueError(f'Unknown checkpoint table: {table}')
        self.connection.executemany(
            f'INSERT INTO {table}(start,end,data) VALUES (?,?,?)',
            ((value['start'], value['end'], json.dumps(value, ensure_ascii=False, allow_nan=False)) for value in values),
        )
        self.connection.commit()

    def rows(self, table):
        if table not in ('chunks', 'turns', 'words', 'segments'):
            raise ValueError(f'Unknown checkpoint table: {table}')
        order = 'start, end, id' if table in ('words', 'turns') else 'id'
        for (data,) in self.connection.execute(f'SELECT data FROM {table} ORDER BY {order}'):
            yield json.loads(data)

    def words_with_ids(self):
        for identity, data in self.connection.execute('SELECT id, data FROM words ORDER BY start, end, id'):
            yield identity, json.loads(data)

    def update_word(self, identity, word, segment_id):
        self.connection.execute('UPDATE words SET data=?, segment_id=? WHERE id=?',
                                (json.dumps(word, ensure_ascii=False, allow_nan=False), segment_id, identity))
        self.connection.commit()

    def prepare_export(self):
        self.connection.execute('DELETE FROM segments')
        self.connection.execute('UPDATE words SET segment_id=NULL WHERE segment_id IS NOT NULL')
        self.connection.commit()

    def start_segment(self, segment):
        cursor = self.connection.execute('INSERT INTO segments(start,end,data) VALUES (?,?,?)',
                                         (segment['start'], segment['end'], json.dumps(segment)))
        self.connection.commit()
        return cursor.lastrowid

    def finish_segment(self, identity, segment):
        self.connection.execute('UPDATE segments SET end=?, data=? WHERE id=?',
                                (segment['end'], json.dumps(segment), identity))
        self.connection.commit()

    def segments_with_ids(self):
        for identity, data in self.connection.execute('SELECT id, data FROM segments ORDER BY id'):
            yield identity, json.loads(data)

    def segment_words(self, identity):
        for (data,) in self.connection.execute(
                'SELECT data FROM words WHERE segment_id=? ORDER BY start,end,id', (identity,)):
            yield json.loads(data)

    def count(self, table):
        if table not in ('chunks', 'turns', 'words', 'segments'):
            raise ValueError(f'Unknown checkpoint table: {table}')
        return self.connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]

    def close(self):
        self.connection.close()


def write_json_array(file, values, resources):
    file.write('[')
    first = True
    for value in values:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        file.flush()
        resources.check_disk(len(text.encode('utf-8'))+2)
        if not first:
            file.write(',\n')
        file.write(text)
        first = False
    file.write(']')


def export_array(path, values, resources):
    target = Path(str(path)+'.part')
    with target.open('w', encoding='utf-8') as file:
        write_json_array(file, values, resources)
    target.replace(path)
