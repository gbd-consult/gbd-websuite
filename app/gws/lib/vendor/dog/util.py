import os
import re
import sys
import random
import subprocess


class Data:
    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def get(self, k, default=None):
        return vars(self).get(k, default)

    def __getattr__(self, item):
        return None


def to_data(arg):
    if isinstance(arg, Data):
        return Data(**vars(arg))
    return Data(**arg)


def parse_args(argv):
    args = {}
    opt = None
    n = 0

    for a in argv:
        if a.startswith('--'):
            opt = a[2:]
            args[opt] = True
        elif a.startswith('-'):
            opt = a[1:]
            args[opt] = True
        elif opt:
            args[opt] = a
            opt = None
        else:
            args[n] = a
            n += 1

    return args


def read_file(path: str) -> str:
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()


def read_file_b(path: str) -> bytes:
    with open(path, 'rb') as fp:
        return fp.read()


def write_file(path: str, s: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(s)


def write_file_b(path: str, s: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fp:
        fp.write(s)


def abspath(path, base):
    if os.path.isabs(path):
        return path
    path = os.path.join(os.path.dirname(base), path)
    return os.path.abspath(path)


def normpath(path):
    res = []
    if path.startswith('/'):
        res.append('')

    for p in path.split('/'):
        p = p.strip()
        if not p or p == '.':
            continue
        if p == '..':
            if not res:
                return ''
            res.pop()
            continue
        if p.startswith('.'):
            return ''
        res.append(p)

    return '/'.join(res)


def random_string(length):
    a = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    r = random.SystemRandom()
    return ''.join(r.choice(a) for _ in range(length))


_UID_DE_TRANS = {
    ord('ä'): 'ae',
    ord('ö'): 'oe',
    ord('ü'): 'ue',
    ord('ß'): 'ss',
}


def to_uid(x) -> str:
    """Convert a value to an uid (alphanumeric string)."""

    if not x:
        return ''
    x = str(x).lower().strip().translate(_UID_DE_TRANS)
    x = re.sub(r'[^a-z0-9]+', '-', x)
    return x.strip('-')


def flatten(ls):
    res = []
    for x in ls:
        if isinstance(x, list):
            res.extend(flatten(x))
        else:
            res.append(x)
    return res


def run(cmd, pipe=False):
    """Run a command, return a tuple (ok, output)"""

    log.debug('run ' + repr(' '.join(cmd)))

    args = {
        'stdout': subprocess.PIPE if pipe else None,
        'stderr': subprocess.STDOUT if pipe else None,
        'stdin': None,
        'shell': False,
    }

    try:
        p = subprocess.Popen(cmd, **args)
        out, _ = p.communicate(None)
        rc = p.returncode
        if isinstance(out, bytes):
            out = out.decode('utf8')
    except Exception as exc:
        log.error(f'run failed: {cmd!r} {exc!r}')
        return False, repr(exc)

    if rc > 0:
        log.error(f'run failed: {cmd!r} {out!r}')
        return False, out

    return True, out


color = {
    'none': '',
    'black': '\u001b[30m',
    'red': '\u001b[31m',
    'green': '\u001b[32m',
    'yellow': '\u001b[33m',
    'blue': '\u001b[34m',
    'magenta': '\u001b[35m',
    'cyan': '\u001b[36m',
    'white': '\u001b[37m',
    'reset': '\u001b[0m',
}

log_colors = {
    'ERROR': 'red',
    'WARNING': 'yellow',
    'INFO': 'green',
    'DEBUG': 'cyan',
}


class _Logger:
    level = 'INFO'

    def set_level(self, level):
        self.level = level

    def log(self, level, *args):
        levels = 'ERROR', 'WARNING', 'INFO', 'DEBUG'
        if levels.index(level) <= levels.index(self.level):
            cin = cout = ''
            if sys.stdout.isatty():
                cin = color[log_colors[level]]
                cout = color['reset']
            a = ' '.join(str(a) for a in args)
            sys.stdout.write(f'{cin}[dog] {level}: {a}{cout}\n')
            sys.stdout.flush()

    def error(self, *args):
        self.log('ERROR', *args)

    def warn(self, *args):
        self.log('WARNING', *args)

    def info(self, *args):
        self.log('INFO', *args)

    def debug(self, *args):
        self.log('DEBUG', *args)


log = _Logger()


def bold(c):
    return c.replace('m', ';1m')


def cprint(clr, msg):
    print(color[clr] + msg + color['reset'])
