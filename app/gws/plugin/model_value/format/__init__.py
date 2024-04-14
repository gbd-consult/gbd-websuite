"""Format value.

This value is computed by applying python `format` to feature attributes.
"""

import gws
import gws.base.model.value
import gws.types as t

gws.ext.new.modelValue('format')


class Config(gws.base.model.value.Config):
    format: str


class Object(gws.base.model.value.Object):
    format: str

    def configure(self):
        self.format = self.cfg('format')

    def compute(self, field, feature, mc):
        return gws.u.format_map(self.format, feature.attributes)
