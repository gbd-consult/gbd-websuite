"""Handle dynamic imports"""

import sys
import os
import importlib

import gws


class Error(gws.Error):
    pass


def import_from_path(path, base_dir=gws.c.APP_DIR):
    abs_path = _abs_path(path, base_dir)
    if not os.path.isfile(abs_path):
        raise Error(f'{abs_path!r}: not found')

    if abs_path.startswith(base_dir):
        # our own module, import relatively to base_dir
        return _do_import(abs_path, base_dir)

    # plugin module, import relatively to the bottom-most "namespace" dir (without __init__)

    dirs = abs_path.strip('/').split('/')
    dirs.pop()

    for n in range(len(dirs), 0, -1):
        ns_dir = '/' + '/'.join(dirs[:n])
        if not os.path.isfile(ns_dir + '/__init__.py'):
            return _do_import(abs_path, ns_dir)

    raise Error(f'{abs_path!r}: cannot locate a base directory')


def _abs_path(path, base_dir):
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    path = os.path.normpath(path)
    if os.path.isdir(path):
        path += '/__init__.py'
    return path


def _do_import(abs_path, base_dir):
    mod_name = _module_name(abs_path[len(base_dir):])

    if mod_name in sys.modules:
        mpath = getattr(sys.modules[mod_name], '__file__', None)
        if mpath != abs_path:
            raise Error(f'{abs_path!r}: overwriting {mod_name!r} from {mpath!r}')
        return sys.modules[mod_name]

    gws.log.debug(f'import: {abs_path=} {mod_name=} {base_dir=}')

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    try:
        return importlib.import_module(mod_name)
    except Exception as exc:
        raise Error(f'{abs_path!r}: import failed') from exc


def _module_name(path):
    parts = path.strip('/').split('/')
    if parts[-1] == '__init__.py':
        parts.pop()
    elif parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]
    return '.'.join(parts)
