"""Current timestamp value."""

import gws
import gws.base.model.value
import gws.lib.datetimex


@gws.ext.config.modelValue('currentTimestamp')
class Config(gws.base.model.value.Config):
    """Value set to the current date and time."""

    pass


@gws.ext.object.modelValue('currentTimestamp')
class Object(gws.base.model.value.Object):
    def compute(self, field, feature, mc):
        return gws.lib.datetimex.now()
