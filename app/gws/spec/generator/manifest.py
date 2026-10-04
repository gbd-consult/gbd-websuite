"""Reader for application manifest files (``MANIFEST.json``)."""

import json
import os


class Error(Exception):
    """Raised when a manifest cannot be parsed."""

    pass


def from_path(path):
    """Read a manifest file.

    Args:
        path: Path to the manifest file.

    Returns:
        A dict with the manifest keys; missing keys get defaults.

    Raises:
        ``Error``: If the manifest is not valid JSON or a value is invalid.
    """

    with open(path, 'rt', encoding='utf8') as fp:
        return from_text(fp.read(), path)


def from_text(text, path):
    """Parse a manifest from text.

    Lines starting with ``//`` or ``#`` are ignored. Relative plugin and
    ``tsConfig`` paths are resolved against the directory of ``path``.

    Args:
        text: Manifest JSON text.
        path: Path to the manifest file.

    Returns:
        A dict with the manifest keys; missing keys get defaults.

    Raises:
        ``Error``: If the text is not valid JSON or a value is invalid.
    """

    lines = []
    for s in text.split('\n'):
        # we allow // or # comments in json
        if s.strip().startswith(('//', '#')):
            s = ''
        lines.append(s)

    try:
        dct = json.loads('\n'.join(lines))
    except Exception as exc:
        raise Error('invalid json') from exc

    return _parse(dct, path)


##

def _version(val, dct, path):
    p = [int(s) for s in val.split('.')]
    return '.'.join(str(s) for s in p)


def _plugins(val, dct, path):
    plugins = []

    for p in val:
        plugin_path = _relpath(p['path'], dct, path)
        plugin_name = p.get('name') or os.path.basename(plugin_path)
        plugins.append({'name': plugin_name, 'path': plugin_path})

    return plugins


def _relpath(val, dct, path):
    basedir = os.path.dirname(path)
    return val if os.path.isabs(val) else os.path.abspath(os.path.join(basedir, val))


def _strlist(val, dct, path):
    return [str(s) for s in val]


def _str(val, dct, path):
    return str(val)


def _bool(val, dct, path):
    return bool(val)


_KEYS = [
    ('uid', _str, None),
    ('release', _version, None),
    ('locales', _strlist, []),
    ('plugins', _plugins, []),
    ('tsConfig', _relpath, ''),
    ('excludePlugins', _strlist, []),
    ('withFallbackConfig', _bool, False),
    ('withStrictConfig', _bool, False),
]


def _parse(dct, path):
    res = {}

    for key, fn, default in _KEYS:
        if key not in dct:
            res[key] = default
        else:
            try:
                res[key] = fn(dct[key], dct, path)
            except Exception as exc:
                raise Error(f'invalid value for key {key!r} in {path!r}') from exc

    return res
