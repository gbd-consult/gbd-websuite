"""Handle dynamic imports"""

import sys
import os
import importlib

import gws


def import_from_path(path, base_dir=gws.APP_DIR):
    in_path, root, mod = _find_import_root_and_module_name(path, base_dir)
    if not in_path:
        sys.path.insert(0, root)
    return importlib.import_module(mod)


def _find_import_root_and_module_name(path, base_dir):
    if not os.path.isabs(path):
        path = base_dir + '/' + path

    init = '__init__.py'
    path = os.path.normpath(path)

    if os.path.isdir(path):
        path += '/' + init
    if not os.path.isfile(path):
        raise ValueError(f'import_from_path: {path!r}: not found')

    dirname, filename = os.path.split(path)
    lastmod = [] if filename == init else [filename.split('.')[0]]
    dirs = dirname.strip('/').split('/')

    # first, try to locate the longest directory root in sys.path

    for n in range(len(dirs), 0, -1):
        root = '/' + '/'.join(dirs[:n])
        if root in sys.path:
            return True, root, '.'.join(dirs[n:] + lastmod)

    # second, find the longest path that doesn't contain __init__.py
    # this will be a new root

    for n in range(len(dirs), 0, -1):
        root = '/' + '/'.join(dirs[:n])
        if not os.path.isfile(root + '/' + init):
            return False, root, '.'.join(dirs[n:] + lastmod)

    raise ValueError(f'import_from_path: {path!r}: cannot be imported')
