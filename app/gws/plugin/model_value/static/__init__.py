"""Static value.

Returns a fixed configured value. Often used with ``isDefault`` to provide a
default for new features, or as a bound of a range validator.

Example::

    fields+ {
        name "status"
        type "text"
        values+ { type "static" value "new" isDefault true forRead false }
    }
"""

from typing import Any


import gws
import gws.base.model.value


@gws.ext.config.modelValue('static')
class Config(gws.base.model.value.Config):
    """Fixed value."""

    value: Any
    """Static value to return."""


@gws.ext.object.modelValue('static')
class Object(gws.base.model.value.Object):
    """Static value object."""

    def compute(self, field, feature, mc):
        return self.cfg('value')
