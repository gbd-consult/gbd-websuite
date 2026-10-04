"""Validator for date ranges.

Checks that the field value is a date between ``min`` and ``max``, both
inclusive. The bounds are model values, so they can be fixed dates or
computed, e.g. the current date. A bound that computes to a string is
compared as an ISO date string. A value that is not a date fails the check.

Example::

    fields+ {
        name "inspection_date"
        type "date"
        validators+ {
            type "dateRange"
            min { type "static" value "2000-01-01" }
            max { type "currentTimestamp" }
        }
    }
"""

from typing import Optional

import gws
import gws.base.model.validator
import gws.lib.datetimex as dt


@gws.ext.config.modelValidator('dateRange')
class Config(gws.base.model.validator.Config):
    """Checks that a date lies within a range."""

    min: Optional[gws.ext.config.modelValue]
    """Earliest allowed date."""
    max: Optional[gws.ext.config.modelValue]
    """Latest allowed date."""


@gws.ext.object.modelValidator('dateRange')
class Object(gws.base.model.validator.Object):
    """Date range validator object."""

    minVal: Optional[gws.ModelValue]
    """Value that computes the earliest allowed date."""
    maxVal: Optional[gws.ModelValue]
    """Value that computes the latest allowed date."""

    def configure(self):
        self.minVal = self.create_child_if_configured(gws.ext.object.modelValue, self.cfg('min'))
        self.maxVal = self.create_child_if_configured(gws.ext.object.modelValue, self.cfg('max'))

    def validate(self, field, feature, mc):
        val = feature.attributes.get(field.name)
        if not dt.is_date(val):
            return False

        d = dt.to_iso_date_string(val)

        if self.minVal:
            v = self.minVal.compute(field, feature, mc)
            s = v if isinstance(v, str) else dt.to_iso_date_string(v)
            if d < s:
                return False

        if self.maxVal:
            v = self.maxVal.compute(field, feature, mc)
            s = v if isinstance(v, str) else dt.to_iso_date_string(v)
            if d > s:
                return False

        return True
