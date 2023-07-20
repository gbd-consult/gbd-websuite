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


def format_map(fmt, data, default=''):
    return fmt.format_map(_Default(data, default))


def retry(times=100, pause=10, factor=1.0):
    return _Retry(times, pause, factor)


def utime():
    return time.time()
