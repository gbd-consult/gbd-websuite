import gws
from . import error, gwsroot


def root() -> gwsroot.Object:
    def _err():
        raise error.LoadError('no configuration root found')

    return gws.get_global('_tree_root', _err)
