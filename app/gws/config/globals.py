import gws

import gws.types as t

from . import error


def root() -> t.IRootObject:
    def _err():
        raise error.LoadError('no configuration root found')
    return gws.get_global('_tree_root', _err)

