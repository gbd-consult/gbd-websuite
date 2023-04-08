"""Static value."""

import gws
import gws.base.model.value
import gws.types as t

gws.ext.new.modelValue('static')


class Config(gws.base.model.value.Config):
    value: t.Any


class Object(gws.base.model.value.Object):
    def evaluate(self, feature, user, **kwargs):
        return self.cfg('value')
