"""Validator for correct values.

When some value (e.g. an integer) cannot be parsed by the field object
(see ``prop_to_py`` methods), it becomes ``gws.ErrorValue``.
This validator checks for this before writing such value is attempted.
"""

import gws
import gws.base.model.validator

gws.ext.new.modelValidator('format')


class Config(gws.base.model.validator.Config):
    pass


class Object(gws.base.model.validator.Object):
    def validate(self, feature, field, user, **kwargs):
        val = feature.attributes.get(field.name)
        return val is not gws.ErrorValue
