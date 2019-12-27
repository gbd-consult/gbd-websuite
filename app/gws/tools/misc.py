import importlib
import importlib.machinery
import importlib.util
import os
import re
import time
import hashlib

import gws


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
    ps = []
    for c in path.split('/'):
        ps.append(c)
        pth = '/'.join(ps)
        if not pth:
            continue
        if not os.path.isdir(pth):
            os.mkdir(pth, mode)
    os.chown(path, gws.UID, gws.GID)
    return path


def running_in_container():
    # see install/build.py
    try:
        return os.path.isfile('/.GWS_IN_CONTAINER')
    except:
        return False


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


def md5(s):
    return hashlib.md5(gws.as_bytes(s)).hexdigest()


def load_source(path, name):
    # see https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


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


# empty image files

class Pixels:
    png8 = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x03\x00\x00\x00(\xcb4\xbb\x00\x00\x00\x06PLTE\xff\xff\xff\x00\x00\x00U\xc2\xd3~\x00\x00\x00\x01tRNS\x00@\xe6\xd8f\x00\x00\x00\x0cIDATx\xdab`\x00\x080\x00\x00\x02\x00\x01OmY\xe1\x00\x00\x00\x00IEND\xaeB`\x82'
    png24 = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x10IDATx\xdab\xf8\xff\xff?\x03@\x80\x01\x00\x08\xfc\x02\xfe\xdb\xa2M\x16\x00\x00\x00\x00IEND\xaeB`\x82'
    jpegBlack = b'\xff\xd8\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xdb\x00C\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0b\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x11\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00?\xf0\x7f\xff\xd9'
    jpegWhite = b'\xff\xd8\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xdb\x00C\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x11\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\x7f\x00\xff\xd9'
    gif = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
