"""Current timestamp value."""

import gws
import gws.base.model.value
import gws.lib.datetime

gws.ext.new.modelValue('currentTimestamp')


class Config(gws.base.model.value.Config):
    pass


class Object(gws.base.model.value.Object):
    def compute(self, field, feature, mc):
        return gws.lib.datetime.now()
