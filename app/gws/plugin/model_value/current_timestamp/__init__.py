"""Current timestamp value.

Computes the current date and time (``gws.lib.datetimex.now``). Typically used
to record when a feature was created or changed, or as a bound of a date range
validator.

Example::

    fields+ {
        name "updated_at"
        type "datetime"
        values+ { type "currentTimestamp" forRead false }
    }
"""

import gws
import gws.base.model.value
import gws.lib.datetimex


@gws.ext.config.modelValue('currentTimestamp')
class Config(gws.base.model.value.Config):
    """Value set to the current date and time."""

    pass


@gws.ext.object.modelValue('currentTimestamp')
class Object(gws.base.model.value.Object):
    """Current timestamp value object."""

    def compute(self, field, feature, mc):
        return gws.lib.datetimex.now()
