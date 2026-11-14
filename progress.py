"""Compact terminal progress; redirected output contains only start/end rows."""
import sys
import threading
import time


class Progress:
    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.tty = self.stream.isatty()
        self.label = None
        self.last = 0
        self.lock = threading.RLock()

    def section(self, name):
        self.stream.write(f'\n{name}:\n'+ '-'*(len(name)+1)+'\n')
        self.stream.flush()

    def start(self, label, status='progress 0%'):
        with self.lock:
            self.label = label
            self.last = 0
            self._write(status)

    def _write(self, status, final=False):
        if self.tty:
            self.stream.write(f'\r\033[2K  {self.label}: {status}'+ ('\n' if final else ''))
        else:
            self.stream.write(f'  {self.label}: {status}\n')
        self.stream.flush()

    def update(self, fraction, prefix='progress'):
        with self.lock:
            if self.label and self.tty and time.monotonic()-self.last >= .15:
                self._write(f'{prefix} {min(99, max(0, int(fraction*100)))}%')
                self.last = time.monotonic()

    def status(self, text):
        with self.lock:
            if self.label and self.tty:
                self._write(text)

    def finish(self, status='done'):
        with self.lock:
            if self.label:
                self._write(status, final=True)
                self.label = None
