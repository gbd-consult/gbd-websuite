"""Validator for non-empty values.

Fails if the field value is None or a string that is empty or consists only of
whitespace. Fields with ``isAuto`` always pass when a feature is created,
since their value is set by the database.

Every field runs a default instance of this validator; a failure is reported
only if the field is required (``isRequired``). A configured ``notEmpty``
validator replaces the default one, e.g. to set a custom message.

Example::

    fields+ {
        name "name"
        type "text"
        isRequired true
        validators+ { type "notEmpty" message "Name is required" }
    }
"""

import gws
import gws.base.model.validator


@gws.ext.config.modelValidator('notEmpty')
class Config(gws.base.model.validator.Config):
    """Checks that a value is not empty."""

    pass


@gws.ext.object.modelValidator('notEmpty')
class Object(gws.base.model.validator.Object):
    """Not-empty validator object."""

    def validate(self, field, feature, mc):
        val = feature.attributes.get(field.name)

        if mc.op == gws.ModelOperation.create and field.isAuto:
            return True
        if isinstance(val, str):
            return len(val.strip()) > 0
        if val is not None:
            return True

        return False
