import json
import os


class Error(Exception):
    pass


def from_path(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return from_text(fp.read(), path)


def from_text(text, path):
    # we allow newline + // or # comments

    lines = []
    for s in text.split('\n'):
        if s.strip().startswith(('//', '#')):
            s = ''
        lines.append(s)

    try:
        js = json.loads('\n'.join(lines))
    except Exception as exc:
        raise Error('invalid json') from exc

    return _parse(js, path)


##

_KEYS = [
    ('release', '_release', None),
    ('customerKey', '_str', ''),
    ('locales', '_strlist', []),
    ('plugins', '_plugins', []),
    ('excludePlugins', '_strlist', []),
    ('withFallbackConfig', '_bool', False),
    ('withStrictConfig', '_bool', False),
]


def _parse(js, path):
    res = {}

    for key, fn, default in _KEYS:
        if key not in js:
            res[key] = default
        else:
            try:
                res[key] = globals()[fn](js[key], js, path)
            except Exception as exc:
                raise Error(f'invalid format for key {key!r}') from exc

    return res


def _release(val, js, path):
    return tuple(int(s) for s in val.split('.'))


def _plugins(val, js, path):
    plugins = []
    dir = os.path.dirname(path)

    for p in val:
        path = p['path'] if os.path.isabs(p['path']) else os.path.abspath(os.path.join(dir, p['path']))
        name = p.get('name') or os.path.basename(path)
        plugins.append({'name': name, 'path': path})

    return plugins


def _strlist(val, js, path):
    return [str(s) for s in val]


def _str(val, js, path):
    return str(val)


def _bool(val, js, path):
    return bool(val)
