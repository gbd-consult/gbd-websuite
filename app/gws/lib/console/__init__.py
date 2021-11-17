import math
import sys
import time

import gws
import gws.types as t


class ProgressIndicator:
    def __init__(self, title, total, resolution=10):
        self.resolution = resolution
        self.title = title
        self.total = total
        self.progress = 0
        self.lastd = 0

    def __enter__(self):
        self.log(f'START (total={self.total})')
        self.starttime = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            ts = time.time() - self.starttime
            self.log(f'END (time={ts:.2f}) sec')

    def update(self, add=1):
        self.progress += add
        p = math.floor(self.progress * 100.0 / self.total)
        if p > 100:
            p = 100
        d = round(p / self.resolution) * self.resolution
        if d > self.lastd:
            self.log(f'{d}%')
        self.lastd = d

    def log(self, s):
        gws.log.info(f'{self.title}: {s}')


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
