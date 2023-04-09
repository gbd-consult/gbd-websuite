import os
import re
import json

from gws.lib.cli import (
    find_files,
    find_dirs,
    read_file,
    write_file,
)


def _json(x):
    if isinstance(x, bytes):
        return x.hex()
    try:
        return vars(x)
    except:
        return repr(x)


def write_json(path, obj):
    write_file(path, json.dumps(obj, default=_json, indent=4, sort_keys=True))


def parse_ini(dct, text):
    section = ''

    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith((';', '#', '//')):
            continue
        if ln[0] == '[':
            section = ln[1:-1].strip()
            continue
        if '=' not in ln:
            raise ValueError(f'invalid ini string {ln!r}')
        key, _, val = ln.partition('=')
        dct.setdefault(section, {})[key.strip()] = val.strip().replace('\\n', '\n')

    return dct


def make_ini(dct):
    buf = []

    for sec, rows in dct.items():
        buf.append('[' + sec + ']')
        for k, v in sorted(rows.items()):
            buf.append(k + '=' + v.replace('\n', '\\n'))
        buf.append('')

    return '\n'.join(buf)
