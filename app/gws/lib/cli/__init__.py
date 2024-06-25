"""Utilities for CLI commands."""

import re
import os
import sys
import subprocess

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

    sys.exit(main_fn(args))
