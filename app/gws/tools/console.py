import sys
import time
import math

import gws


class ProgressIndicator:
    def __init__(self, title, total, resolution=10):
        self.isatty = sys.stderr.isatty()
        self.resolution = resolution
        self.title = title
        self.total = total
        self.progress = 0
        self.lastd = 0

    def __enter__(self):
        self.write('START (total=%d)' % self.total)
        self.starttime = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.write('\n')
        else:
            t = time.time() - self.starttime
            self.write('END (time=%.2f rps=%.1f)' % (t, self.total / t))

    def update(self, add=1):
        self.progress += add
        p = math.floor(self.progress * 100.0 / self.total)
        if p > 100:
            p = 100
        d = round(p / self.resolution) * self.resolution
        if d > self.lastd:
            self.write(str(d) + '%%')
        self.lastd = d

    def write(self, s):
        gws.log.info(self.title + ': ' + s, stacklevel=2)
