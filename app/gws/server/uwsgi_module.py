import importlib
import sys


def load():
    name = 'uwsgi'
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)
