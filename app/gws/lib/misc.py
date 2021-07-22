import importlib
import importlib.machinery
import importlib.util
import os
import time


class _Retry(object):
    def __init__(self, times, pause, factor):
        self.times = times - 1
        self.pause = pause
        self.factor = factor
        self.start = time.time()

    def __iter__(self):
        while self.times >= 0:
            yield self
            time.sleep(self.pause)
            self.times -= 1
            self.pause *= self.factor

    def __repr__(self):
        return f'(retry={self.times})'

    def __nonzero__(self):
        return self.times


class _Default:
    def __init__(self, d, default):
        self.d = d
        self.default = default

    def __getitem__(self, item):
        t = self.d.get(item)
        return self.default if t is None else t


def format_placeholders(fmt, data, default=''):
    return fmt.format_map(_Default(data, default))


def retry(times=100, pause=10, factor=1.0):
    return _Retry(times, pause, factor)


def utime():
    return time.time()


class lock:
    def __init__(self, path):
        self.path = path
        self.ok = False

    def __enter__(self):
        try:
            fp = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fp)
            self.ok = True
        except:
            self.ok = False
        return self.ok

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ok:
            try:
                os.unlink(self.path)
            except OSError:
                pass


def text_table(data, header=None, delim=' | '):
    """Format a list of dicts as a text-mode table."""

    data = list(data)

    if not data:
        return ''

    print_header = header is not None
    if header is None or header == 'auto':
        header = sorted(data[0].keys())

    widths = [len(h) if print_header else 1 for h in header]

    for d in data:
        widths = [
            max(a, b)
            for a, b in zip(
                widths,
                [len(str(d.get(h, ''))) for h in header]
            )
        ]

    def field(n, v):
        if isinstance(v, (int, float)):
            return str(v).rjust(widths[n])
        return str(v).ljust(widths[n])

    rows = []

    if print_header:
        hdr = delim.join(field(n, h) for n, h in enumerate(header))
        rows.append(hdr)
        rows.append('-' * len(hdr))

    for d in data:
        rows.append(delim.join(field(n, d.get(h, '')) for n, h in enumerate(header)))

    return '\n'.join(rows)
