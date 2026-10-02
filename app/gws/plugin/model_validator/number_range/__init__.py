"""Validator for number ranges."""

from typing import Optional

import gws
import gws.base.model.validator


@gws.ext.config.modelValidator('numberRange')
class Config(gws.base.model.validator.Config):
    """Checks that a number lies within a range."""

    min: Optional[gws.ext.config.modelValue]
    """Smallest allowed value."""
    max: Optional[gws.ext.config.modelValue]
    """Largest allowed value."""


@gws.ext.object.modelValidator('numberRange')
class Object(gws.base.model.validator.Object):
    minVal: Optional[gws.ModelValue]
    maxVal: Optional[gws.ModelValue]

    def configure(self):
        self.minVal = self.create_child_if_configured(gws.ext.object.modelValue, self.cfg('min'))
        self.maxVal = self.create_child_if_configured(gws.ext.object.modelValue, self.cfg('max'))

    def validate(self, field, feature, mc):
        val = feature.attributes.get(field.name)
        if not isinstance(val, (int, float)):
            return False

        if self.minVal:
            v = self.minVal.compute(field, feature, mc)
            if val < v:
                return False

        if self.maxVal:
            v = self.maxVal.compute(field, feature, mc)
            if val > v:
                return False

        return True
