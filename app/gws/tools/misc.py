import json
import importlib
import importlib.machinery
import importlib.util
import os
import re
import time
import hashlib

import gws

# OGC's 1px = 0.28mm
# https://portal.opengeospatial.org/files/?artifact_id=14416 page 27

OGC_M_PER_PX = 0.00028
OGC_SCREEN_PPI = 25.4 / OGC_M_PER_PX / 1000

PDF_DPI = 96

MM_PER_IN = 25.4
PT_PER_IN = 72


def scale2res(scale):
    return scale * OGC_M_PER_PX


def res2scale(resolution):
    return resolution / OGC_M_PER_PX


def mm2in(x):
    return x / MM_PER_IN


def m2in(x):
    return (x / MM_PER_IN) * 1000


def in2mm(x):
    return x * MM_PER_IN


def in2m(x):
    return (x * MM_PER_IN) / 1000


def in2px(x, ppi):
    return x * ppi


def mm2px(x, ppi):
    return int((x * ppi) / MM_PER_IN)


def px2mm(x, ppi):
    return int((x / ppi) * MM_PER_IN)


def mm2pt(x):
    return (x / MM_PER_IN) * PT_PER_IN


def pt2mm(x):
    return (x / PT_PER_IN) * MM_PER_IN


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


def find_files(dirname, pattern=None):
    for fname in os.listdir(dirname):
        if fname.startswith('.'):
            continue

        path = os.path.join(dirname, fname)

        if os.path.isdir(path):
            yield from find_files(path, pattern)
            continue

        if pattern is None or re.search(pattern, path):
            yield path


def ensure_dir(path, base=None, mode=0o755):
    if base:
        path = os.path.join(base, path)
    os.makedirs(path, mode, exist_ok=True)
    return path


def parse_path(path):
    """Parse a path into a dict(path,dirname,filename,name,extension)"""

    d = {'path': path}

    d['dirname'], d['filename'] = os.path.split(path)
    if d['filename'].startswith('.'):
        d['name'], d['extension'] = d['filename'], ''
    else:
        d['name'], _, d['extension'] = d['filename'].partition('.')

    return d


def sha256(s):
    return hashlib.sha256(gws.as_bytes(s)).hexdigest()


def load_source(path, name):
    # see https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


_unit_re = re.compile(r'''(?x)
    ^
        (?P<number>
            -?
            (\d+ (\.\d*)? )
            |
            (\.\d+)
        )
        (?P<rest> .*)
    $
''')


def parse_unit(s, unit=None):
    if isinstance(s, (int, float)):
        if not unit:
            raise ValueError('parse_unit: unit required', s)
        n = float(s)
        u = gws.as_str(unit).lower()
    else:
        s = gws.as_str(s).strip()
        m = _unit_re.match(s)
        if not m:
            raise ValueError('parse_unit: not a number', s)
        n = float(m.group('number'))
        u = (m.group('rest').strip() or unit).lower()

    if u == 'm':
        u = 'mm'
        n *= 1000

    elif u == 'cm':
        u = 'mm'
        n *= 100

    if u not in ('mm', 'in', 'px'):
        raise ValueError('parse_unit: invalid unit', s)

    return n, u


_durations = {
    'w': 3600 * 24 * 7,
    'd': 3600 * 24,
    'h': 3600,
    'm': 60,
    's': 1,
}


def parse_duration(s):
    if isinstance(s, int):
        return s

    p = None
    r = 0

    for n, v in re.findall(r'(\d+)|(\D+)', str(s).strip()):
        if n:
            p = int(n)
            continue
        v = v.strip()
        if p is None or v not in _durations:
            raise ValueError('invalid duration', s)
        r += p * _durations[v]
        p = None

    if p:
        r += p

    return r


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
