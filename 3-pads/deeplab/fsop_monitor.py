"""Filesystem-operation counting for RQ2 (Table~\\ref{tab:rq2-fsops}).

Counts open() calls, os.stat() calls (which also catches os.path.exists() /
isfile(), since posixpath's implementation calls os.stat() through the same
os module object being patched here), and bytes read through open() file
handles, for the duration of a `with` block. This is the direct measure RQ2
needs: sharding is intended to convert many small metadata operations into
few large sequential reads, and that is only verifiable by actually counting
operations, not by inferring it from wall-clock time.

IMPORTANT scoping constraint: this patches builtins.open/os.stat in the
CURRENT PROCESS only. A PyTorch DataLoader with num_workers > 0 runs
__getitem__ in separate worker subprocesses, which do not share this
process's patched builtins -- their file operations would not be counted at
all, silently. For that reason this is wired into profile_data_pipeline()
(train.py), which already builds its own num_workers=0 DataLoader for exactly
this kind of single-process measurement, rather than around the main
multi-worker training DataLoader. Do not reuse this around a DataLoader that
has num_workers > 0; the counts would be wrong (under-counted), not just
imprecise.
"""
import builtins
import os
import threading


class _CountingFile:
    """Wraps a file object opened through the patched open() to count bytes
    read without changing read()/iteration semantics for the caller."""

    def __init__(self, f, counter):
        self._f = f
        self._counter = counter

    def read(self, *args, **kwargs):
        data = self._f.read(*args, **kwargs)
        n = len(data) if data is not None else 0
        with self._counter._lock:
            self._counter.bytes_read += n
        return data

    def readline(self, *args, **kwargs):
        data = self._f.readline(*args, **kwargs)
        n = len(data) if data is not None else 0
        with self._counter._lock:
            self._counter.bytes_read += n
        return data

    def __getattr__(self, name):
        return getattr(self._f, name)

    def __enter__(self):
        self._f.__enter__()
        return self

    def __exit__(self, *exc):
        return self._f.__exit__(*exc)

    def __iter__(self):
        return iter(self._f)


class FSOpCounter:
    """Usage:
        with FSOpCounter() as counter:
            ... single-process (num_workers=0) file access ...
        print(counter.summary())
    """

    def __init__(self):
        self.open_calls = 0
        self.stat_calls = 0
        self.bytes_read = 0
        self._lock = threading.Lock()
        self._orig_open = None
        self._orig_stat = None

    def _counting_open(self, file, mode="r", *args, **kwargs):
        with self._lock:
            self.open_calls += 1
        f = self._orig_open(file, mode, *args, **kwargs)
        if "r" in mode:
            return _CountingFile(f, self)
        return f

    def _counting_stat(self, path, *args, **kwargs):
        with self._lock:
            self.stat_calls += 1
        return self._orig_stat(path, *args, **kwargs)

    def start(self):
        self._orig_open = builtins.open
        self._orig_stat = os.stat
        builtins.open = self._counting_open
        os.stat = self._counting_stat
        return self

    def stop(self):
        if self._orig_open is not None:
            builtins.open = self._orig_open
            self._orig_open = None
        if self._orig_stat is not None:
            os.stat = self._orig_stat
            self._orig_stat = None

    def summary(self, batches=None):
        result = {
            "open_calls": self.open_calls,
            "stat_calls": self.stat_calls,
            "bytes_read": self.bytes_read,
        }
        if batches:
            result["open_calls_per_batch"] = self.open_calls / batches
            result["stat_calls_per_batch"] = self.stat_calls / batches
            result["bytes_read_per_batch"] = self.bytes_read / batches
        return result

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
