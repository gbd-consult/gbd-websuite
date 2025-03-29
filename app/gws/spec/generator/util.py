import os
import re
import json

from gws.lib.cli import (
    find_files,
    find_dirs,
    read_file,
    write_file,
    ensure_dir,
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


def parse_ini(text):
    dct = {}
    section = ''
    key = ''

    for ln in text.strip().splitlines():
        ln = ln.strip()
        if ln.startswith((';', '#', '//')):
            continue
        if ln.startswith('['):
            section = ln[1:-1].strip()
            continue
        m = re.match(r'^([a-zA-Z0-9_.]+)\s*=(.*)', ln)
        if m:
            key = m.group(1).strip()
            val = m.group(2)
            dct.setdefault(section, {})[key] = val.strip().replace('\\n', '\n')
        elif key:
            dct[section][key] += '\n' + ln.strip()
        elif ln:
            raise ValueError(f'invalid ini string {ln!r}')

    return dct


def make_ini(dct):
    buf = []

    for sec, rows in dct.items():
        buf.append('[' + sec + ']')
        for k, v in sorted(rows.items()):
            buf.append(k + '=' + v.replace('\n', '\\n'))
        buf.append('')

    return '\n'.join(buf)
