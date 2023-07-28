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
    def compute(self, feature, field, user, **kwargs):
        return gws.format_map(self.cfg('format'), feature.attributes)
