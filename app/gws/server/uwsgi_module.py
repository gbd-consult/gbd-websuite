"""Access to the ``uwsgi`` module."""

import importlib
import sys


def load():
    """Return the ``uwsgi`` module, which is only available inside a uWSGI process.

    Returns:
        The ``uwsgi`` module.

    Raises:
        ``ModuleNotFoundError``: If not running in uWSGI.
    """
    name = 'uwsgi'
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)
