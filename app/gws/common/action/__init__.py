"""Generic action object, the parent of all action objects."""

import gws
import gws.types as t


class Object(gws.Object):
    type: str = ''

    def configure(self):
        super().configure()
        self.type = self.var('type')
