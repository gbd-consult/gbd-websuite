import gws
import gws.types as t
from . import error, gwsroot


def root() -> gwsroot.Object:
    def _err():
        raise error.LoadError('no configuration root found')

    return gws.get_global('_tree_root', _err)


def app_data():
    return root().app_data


def find(klass, uid) -> t.ObjectInterface:
    return root().find(klass, uid)


def find_first(klass) -> t.ObjectInterface:
    return root().find_first(klass)


def find_all(klass) -> t.List[t.ObjectInterface]:
    return root().find_all(klass)


def var(key=None, default=None):
    return root().var(key, default)
