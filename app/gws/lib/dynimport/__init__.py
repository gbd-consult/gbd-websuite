"""Handle dynamic imports"""

import sys
import os
import importlib

import gws


class Error(gws.Error):
    """Custom error class for import-related exceptions."""
    pass


def load_file(path: str) -> dict:
    """Load a python file and return its globals."""

    return load_string(gws.u.read_file(path), path)


def load_string(text: str, path='') -> dict:
    """Load a string as python code and return its globals."""

    globs = {'__file__': path}
    code = compile(text, path, 'exec')
    exec(code, globs)
    return globs


def import_from_path(path: str, base_dir: str = gws.c.APP_DIR):
    """Imports a module from a given file path.

    Args:
        path: The relative or absolute path to the module file.
        base_dir: The base directory to resolve relative paths. Defaults to `gws.c.APP_DIR`.

    Returns:
        The imported module.

    Raises:
        Error: If the module file is not found or a base directory cannot be located.
    """
    abs_path = _abs_path(path, base_dir)
    if not os.path.isfile(abs_path):
        raise Error(f'{abs_path!r}: not found')

    if abs_path.startswith(base_dir):
        # Our own module, import relatively to base_dir
        return _do_import(abs_path, base_dir)

    # Plugin module, import relative to the bottom-most "namespace" dir (without __init__)
    dirs = abs_path.strip('/').split('/')
    dirs.pop()

    for n in range(len(dirs), 0, -1):
        ns_dir = '/' + '/'.join(dirs[:n])
        if not os.path.isfile(ns_dir + '/__init__.py'):
            return _do_import(abs_path, ns_dir)

    raise Error(f'{abs_path!r}: cannot locate a base directory')


def _abs_path(path: str, base_dir: str) -> str:
    """Converts a relative path to an absolute normalized path.

    Args:
        path: The input file path.
        base_dir: The base directory for resolving relative paths.

    Returns:
        The absolute, normalized file path.
    """
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    path = os.path.normpath(path)
    if os.path.isdir(path):
        path += '/__init__.py'
    return path


def _do_import(abs_path: str, base_dir: str):
    """Imports a module given its absolute path and base directory.

    Args:
        abs_path: The absolute path to the module file.
        base_dir: The base directory for resolving module names.

    Returns:
        The imported module.

    Raises:
        Error: If the module import fails or an existing module is being overwritten.
    """
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


def _module_name(path: str) -> str:
    """Derives the module name from a given file path.

    Args:
        path: The file path of the module.

    Returns:
        The module name in dotted notation.
    """
    parts = path.strip('/').split('/')
    if parts[-1] == '__init__.py':
        parts.pop()
    elif parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]
    return '.'.join(parts)
