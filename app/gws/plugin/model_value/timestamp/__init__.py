"""Current timestamp value."""

import gws
import gws.base.model.value
import gws.lib.date
import gws.types as t

gws.ext.new.modelValue('timestamp')


class Config(gws.base.model.value.Config):
    pass


class Object(gws.base.model.value.Object):
    def compute(self, field, feature, mc):
        return gws.lib.date.now()
