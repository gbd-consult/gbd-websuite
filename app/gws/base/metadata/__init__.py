"""Metadata object."""

import gws
import gws.lib.metadata
import gws.types as t


##



##


class Object(gws.Object):
    def __init__(self, md: gws.Metadata):
        self.md = gws.lib.metadata.check(md)


    def extend(self, *others, extend_lists=False):
        self.md = gws.lib.metadata.extend(self.md, *others, extend_lists=extend_lists)
        return self

    def get(self, key, default=None):
        return self.md.get(key, default)

    def set(self, key, val):
        self.md = gws.lib.metadata.set_value(self.md, key, val)
        return self
