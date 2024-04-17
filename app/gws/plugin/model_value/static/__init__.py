"""Static value."""

from typing import Any


import gws
import gws.base.model.value

gws.ext.new.modelValue('static')


class Config(gws.base.model.value.Config):
    value: Any


class Object(gws.base.model.value.Object):
    def compute(self, field, feature, mc):
        return self.cfg('value')
