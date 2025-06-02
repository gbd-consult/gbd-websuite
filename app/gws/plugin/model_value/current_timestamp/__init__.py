"""Current timestamp value."""

import gws
import gws.base.model.value
import gws.lib.datetimex

gws.ext.new.modelValue('currentTimestamp')


class Config(gws.base.model.value.Config):
    """Current timestamp value configuration."""

    pass


class Object(gws.base.model.value.Object):
    def compute(self, field, feature, mc):
        return gws.lib.datetimex.now()
