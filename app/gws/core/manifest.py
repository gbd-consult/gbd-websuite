"""Tools to deal with r8 MANIFEST files."""

import json
import os


class Error(Exception):
    pass


def from_path(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return from_text(fp.read(), path)


def from_text(text, path):
    lines = []
    for s in text.split('\n'):
        # we allow // or # comments in json
        if s.strip().startswith(('//', '#')):
            s = ''
        lines.append(s)

    try:
        js = json.loads('\n'.join(lines))
    except Exception as exc:
        raise Error('invalid json') from exc

    return _parse(js, path)


##

def _version(val, js, path):
    p = [int(s) for s in val.split('.')]
    return '.'.join(str(s) for s in p)


def _plugins(val, js, path):
    plugins = []
    basedir = os.path.dirname(path)

    for p in val:
        path = p['path'] if os.path.isabs(p['path']) else os.path.abspath(os.path.join(basedir, p['path']))
        name = p.get('name') or os.path.basename(path)
        plugins.append({'name': name, 'path': path})

    return plugins


def _strlist(val, js, path):
    return [str(s) for s in val]


def _str(val, js, path):
    return str(val)


def _bool(val, js, path):
    return bool(val)


_KEYS = [
    ('uid', _str, None),
    ('release', _version, None),
    ('locales', _strlist, []),
    ('plugins', _plugins, []),
    ('excludePlugins', _strlist, []),
    ('withFallbackConfig', _bool, False),
    ('withStrictConfig', _bool, False),
]


def _parse(js, path):
    res = {}

    for key, fn, default in _KEYS:
        if key not in js:
            res[key] = default
        else:
            try:
                res[key] = fn(js[key], js, path)
            except Exception as exc:
                raise Error(f'invalid value for key {key!r} in {path!r}') from exc

    return res
