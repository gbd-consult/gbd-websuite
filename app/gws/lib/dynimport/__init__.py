"""Dynamic imports of Python code.

This package loads Python code at run time, either as a plain script or as a regular module:

- ``load_file`` and ``load_string`` execute Python source and return its global namespace.
  Nothing is added to ``sys.modules``. This is used, for example, for Python config files.
- ``import_from_path`` imports a module from a file path and registers it in ``sys.modules``.
  Paths inside the application directory are imported as part of the ``gws`` package.
  Other paths (plugins) are imported relative to the deepest parent directory that is not a package
  (has no ``__init__.py``), so that the plugin's own package structure is preserved.

Example::

    import gws.lib.dynimport

    fn = gws.lib.dynimport.load_file('/data/config.py').get('main')

    mod = gws.lib.dynimport.import_from_path('gws/plugin/ows_client/wms/caps.py')
"""

import sys
import os
import importlib

import gws


class Error(gws.Error):
    """Dynamic import error."""
    pass


def load_file(path: str) -> dict:
    """Execute a Python file and return its globals.

    Args:
        path: Path to the Python file.

    Returns:
        The global namespace of the executed code.
    """

    return load_string(gws.u.read_file(path), path)


def load_string(text: str, path='') -> dict:
    """Execute a string as Python code and return its globals.

    Args:
        text: Python source code.
        path: File path, used for ``__file__`` and in error messages.

    Returns:
        The global namespace of the executed code.
    """

    globs = {'__file__': path}
    code = compile(text, path, 'exec')
    exec(code, globs)
    return globs


def import_from_path(path: str, base_dir: str = gws.c.APP_DIR):
    """Import a module from a file path.

    If the path is a directory, its ``__init__.py`` is imported. If a module with the same name
    is already imported from the same file, it is returned as is.

    Args:
        path: Relative or absolute path to the module file or package directory.
        base_dir: Base directory to resolve relative paths, the application directory by default.

    Returns:
        The imported module.

    Raises:
        ``Error``: If the module file is not found, a base directory cannot be located,
            a module with the same name was imported from a different file, or the import fails.
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
    """Convert a path to an absolute normalized path of a Python file."""
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    path = os.path.normpath(path)
    if os.path.isdir(path):
        path += '/__init__.py'
    return path


def _do_import(abs_path: str, base_dir: str):
    """Import a module by its absolute path, with the module name relative to the base directory."""
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
    """Derive a dotted module name from a relative file path."""
    parts = path.strip('/').split('/')
    if parts[-1] == '__init__.py':
        parts.pop()
    elif parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]
    return '.'.join(parts)
