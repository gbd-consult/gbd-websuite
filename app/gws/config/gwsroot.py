import gws
import gws.core.tree

import gws.types as t

from . import spec, error


class Object(gws.core.tree.RootObject):
    def configure(self):
        super().configure()
        self.validator = spec.validator()
        self.application = t.cast(t.IApplication, self.create_child('gws.common.application', self.config))


def create() -> t.IRootObject:
    return gws.set_global('_tree_root', Object())


def root() -> t.IRootObject:
    def _err():
        raise error.LoadError('no configuration root found')

    return gws.get_global('_tree_root', _err)
