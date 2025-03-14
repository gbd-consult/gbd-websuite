"""Utilities for CLI commands."""

import re
import os
import shutil
import sys
import subprocess
import time
import math
import traceback

SCRIPT_NAME = ''

_COLOR = {
    'black': '\x1b[30m',
    'red': '\x1b[31m',
    'green': '\x1b[32m',
    'yellow': '\x1b[33m',
    'blue': '\x1b[34m',
    'magenta': '\x1b[35m',
    'cyan': '\x1b[36m',
    'white': '\x1b[37m',
    'reset': '\x1b[0m',
}


def cprint(clr, msg):
    if SCRIPT_NAME:
        msg = '[' + SCRIPT_NAME + '] ' + msg
    if clr and sys.stdout.isatty():
        msg = _COLOR[clr] + msg + _COLOR['reset']
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


def error(msg):
    cprint('red', msg)


def fatal(msg):
    cprint('red', msg)
    sys.exit(1)


def warning(msg):
    cprint('yellow', msg)


def info(msg):
    cprint('cyan', msg)


##

def run(cmd):
    if isinstance(cmd, list):
        cmd = ' '.join(cmd)
    cmd = re.sub(r'\s+', ' ', cmd.strip())
    info(f'> {cmd}')
    res = subprocess.run(cmd, shell=True, capture_output=False)
    if res.returncode:
        fatal(f'COMMAND FAILED, code {res.returncode}')


def exec(cmd):
    try:
        return (
            subprocess
            .run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            .stdout.decode('utf8').strip()
        )
    except Exception as exc:
        return f'> {cmd} FAILED: {exc}'


def find_dirs(dirname):
    if not os.path.isdir(dirname):
        return

    de: os.DirEntry
    for de in os.scandir(dirname):
        if de.name.startswith('.'):
            continue
        if de.is_dir():
            yield de.path


def find_files(dirname, pattern=None, deep=True):
    if not os.path.isdir(dirname):
        return

    de: os.DirEntry
    for de in os.scandir(dirname):
        if de.name.startswith('.'):
            continue
        if de.is_dir() and deep:
            yield from find_files(de.path, pattern)
            continue
        if de.is_file() and (pattern is None or re.search(pattern, de.path)):
            yield de.path


def ensure_dir(path, clear=False):
    os.makedirs(path, exist_ok=True)
    if clear:
        shutil.rmtree(path)
    return path


def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read().strip()


def write_file(path, text):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


def parse_args(argv):
    args = {}
    opt = None
    n = 0

    for a in argv:
        if a == '-':
            args['_rest'] = []
        elif '_rest' in args:
            args['_rest'].append(a)
        elif a.startswith('--'):
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


def main(name, main_fn, usage):
    global SCRIPT_NAME

    SCRIPT_NAME = name

    args = parse_args(sys.argv)
    if not args or 'h' in args or 'help' in args:
        print('\n' + usage.strip() + '\n')
        sys.exit(0)

    try:
        sys.exit(main_fn(args))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        error('INTERNAL ERROR')
        error(traceback.format_exc())


def text_table(data, header=None, delim=' | '):
    """Format a list of dicts as a text-mode table."""

    data = list(data)

    if not data:
        return ''

    is_dict = isinstance(data[0], dict)

    print_header = header is not None
    if header is None or header == 'auto':
        header = data[0].keys() if is_dict else list(range(len(data[0])))

    widths = [len(h) if print_header else 1 for h in header]

    def get(d, h):
        if is_dict:
            return d.get(h, '')
        try:
            return d[h]
        except IndexError:
            return ''

    for d in data:
        widths = [
            max(a, b)
            for a, b in zip(
                widths,
                [len(str(get(d, h))) for h in header]
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
        rows.append(delim.join(field(n, get(d, h)) for n, h in enumerate(header)))

    return '\n'.join(rows)


class ProgressIndicator:
    def __init__(self, title, total=0, resolution=10):
        self.resolution = resolution
        self.title = title
        self.total = total
        self.progress = 0
        self.lastd = 0
        self.starttime = 0

    def __enter__(self):
        self.log(f'START ({self.total})' if self.total else 'START')
        self.starttime = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            ts = time.time() - self.starttime
            self.log(f'END ({ts:.2f} sec)')

    def update(self, add=1):
        if not self.total:
            return
        self.progress += add
        p = math.floor(self.progress * 100.0 / self.total)
        if p > 100:
            p = 100
        d = round(p / self.resolution) * self.resolution
        if d > self.lastd:
            self.log(f'{d}%')
        self.lastd = d

    def log(self, s):
        info(f'{self.title}: {s}')
