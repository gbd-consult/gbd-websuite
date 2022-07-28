import os
import re
import json


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


def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read().strip()


def write_file(path, text):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


def _json(x):
    if isinstance(x, bytes):
        return x.hex()
    try:
        return vars(x)
    except:
        return repr(x)


def write_json(path, obj):
    write_file(path, json.dumps(obj, default=_json, indent=4))


