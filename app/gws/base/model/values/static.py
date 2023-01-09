"""Static value."""

import gws
import gws.base.model.value
import gws.types as t


@gws.ext.config.modelValue('static')
class Config(gws.base.model.value.Config):
    value: t.Any


@gws.ext.object.modelValue('static')
class Object(gws.base.model.value.Object):
    def evaluate(self, feature, user, **kwargs):
        return self.var('value')
