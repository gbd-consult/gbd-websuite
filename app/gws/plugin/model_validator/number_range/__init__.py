"""Validator for number ranges.

Checks that the field value is an ``int`` or ``float`` between ``min`` and
``max``, both inclusive. The bounds are model values, so they can be fixed
numbers or computed. A value that is not a number fails the check.

Example::

    fields+ {
        name "floors"
        type "integer"
        validators+ {
            type "numberRange"
            min { type "static" value 1 }
            max { type "static" value 99 }
        }
    }
"""

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
    """Number range validator object."""

    minVal: Optional[gws.ModelValue]
    """Value that computes the smallest allowed number."""
    maxVal: Optional[gws.ModelValue]
    """Value that computes the largest allowed number."""

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
