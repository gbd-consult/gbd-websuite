"""Validator for date ranges."""

from typing import Optional

import gws
import gws.base.model.validator
import gws.lib.datetimex as dt

gws.ext.new.modelValidator('dateRange')


class Config(gws.base.model.validator.Config):
    """Validator for date ranges. (added in 8.1)"""
    min: Optional[gws.ext.config.modelValue]
    max: Optional[gws.ext.config.modelValue]


class Object(gws.base.model.validator.Object):
    minVal: Optional[gws.ModelValue]
    maxVal: Optional[gws.ModelValue]

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
