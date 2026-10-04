"""File, JSON and ini helpers for the generator."""

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
    """Write an object to a JSON file.

    Objects are written as their attribute dicts, bytes as hex strings,
    other values that are not JSON serializable as their ``repr``.

    Args:
        path: File path.
        obj: Object to write.
    """

    write_file(path, json.dumps(obj, default=_json, indent=4, sort_keys=True))


def read_json(path):
    """Read a JSON file.

    Args:
        path: File path.

    Returns:
        The parsed JSON value.
    """

    return json.loads(read_file(path))


def parse_ini(text):
    """Parse an ini-style strings file.

    Lines starting with ``;``, ``#`` or ``//`` are comments. A line without
    ``=`` continues the value of the previous key. ``\\n`` in values is
    converted to a newline.

    Args:
        text: File content.

    Returns:
        A dict of sections, each a dict of keys and values.

    Raises:
        ``ValueError``: If a line cannot be parsed.
    """

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
    """Create an ini-style text from a dict of sections.

    Keys are sorted, newlines in values are written as ``\\n``.

    Args:
        dct: A dict of sections, each a dict of keys and values.

    Returns:
        The ini text.
    """

    buf = []

    for sec, rows in dct.items():
        buf.append('[' + sec + ']')
        for k, v in sorted(rows.items()):
            buf.append(k + '=' + v.replace('\n', '\\n'))
        buf.append('')

    return '\n'.join(buf)
