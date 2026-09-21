"""Static value."""

from typing import Any


import gws
import gws.base.model.value


@gws.ext.config.modelValue('static')
class Config(gws.base.model.value.Config):
    """Static value configuration."""

    value: Any
    """Static value to return."""


@gws.ext.object.modelValue('static')
class Object(gws.base.model.value.Object):
    def compute(self, field, feature, mc):
        return self.cfg('value')
