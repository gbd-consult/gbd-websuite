"""Validator for correctly parsed values.

When a value from the client (e.g. an integer) cannot be parsed by the field,
the field stores ``gws.ErrorValue`` instead. This validator fails for such
values, so that they are reported before the feature is written.

Every field runs a default instance of this validator. A configured
``format`` validator replaces the default one, e.g. to set a custom message.

Example::

    fields+ {
        name "count"
        type "integer"
        validators+ { type "format" message "Please enter a whole number" }
    }
"""

import gws
import gws.base.model.validator


@gws.ext.config.modelValidator('format')
class Config(gws.base.model.validator.Config):
    """Checks that a value could be parsed for the field type."""

    pass


@gws.ext.object.modelValidator('format')
class Object(gws.base.model.validator.Object):
    """Format validator object."""

    def validate(self, field, feature, mc):
        val = feature.attributes.get(field.name)
        return val is not gws.ErrorValue
