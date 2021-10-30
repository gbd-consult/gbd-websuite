"""Parser for manifest files."""

import json
import os
import re

from . import base


def load(path):
    """Load the manifest from the path"""

    with open(path, 'rt', encoding='utf8') as fp:
        text = fp.read()

    # we allow // comments in manifests
    text = re.sub(r'//.*', '', text)

    return _convert_value(path, '', json.loads(text))


def enumerate_plugins(mfst, local_plugin_dir):
    if not mfst:
        return list(_local_plugins(local_plugin_dir))

    lst = mfst.get('plugins')
    if lst is not None:
        return list(_manifest_plugins(lst))

    cdict = {c.name: c for c in _local_plugins(local_plugin_dir)}

    lst = mfst.get('addPlugins')
    if lst is not None:
        cdict.update({c.name: c for c in _manifest_plugins(lst)})

    chunks = list(cdict.values())

    lst = mfst.get('excludePlugins')
    if lst is not None:
        lst = set(_plugin_name(p) for p in lst)
        chunks = [p for p in chunks if p.name not in lst]

    return chunks


def _local_plugins(basedir):
    for path in _find_dirs(basedir):
        name = os.path.basename(path)
        yield base.Data(name=_plugin_name(name), sourceDir=path, bundleDir=path)


def _manifest_plugins(lst):
    for p in lst:
        path = p.get('path')
        name = p.get('name') or os.path.basename(path)
        yield base.Data(name=_plugin_name(name), sourceDir=path, bundleDir=path)


def _plugin_name(name):
    return base.GWS_PLUGIN_PREFIX + '.' + name


def _convert_value(path, key, val):
    if isinstance(val, str):
        val = _replace_env(val)
        if key.lower().endswith('path') and val.startswith('.'):
            val = os.path.abspath(os.path.join(os.path.dirname(path), val))
        return val
    if isinstance(val, list):
        return [_convert_value(path, '', v) for v in val]
    if isinstance(val, dict):
        return {k: _convert_value(path, k, v) for k, v in val.items()}
    return val


def _replace_env(s):
    def _env(m):
        key = m[1]
        if key in os.environ:
            return os.environ[key]
        raise ValueError(f'unknown variable {key!r} in {s!r}')

    return re.sub(r'\${(\w+)}', _env, s)


def _find_dirs(basedir):
    for fname in os.listdir(basedir):
        if fname.startswith('.'):
            continue
        path = os.path.join(basedir, fname)
        if os.path.isdir(path):
            yield path
