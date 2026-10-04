"""Format value.

Computes a string by applying a Python format string to the feature
attributes, using ``gws.u.format_map``.

Example::

    fields+ {
        name "label"
        type "text"
        values+ { type "format" format "{street} {house_number}" }
    }
"""

import gws
import gws.base.model.value


@gws.ext.config.modelValue('format')
class Config(gws.base.model.value.Config):
    """Value computed by a format string over feature attributes."""

    format: str
    """Format string to apply to feature attributes."""


@gws.ext.object.modelValue('format')
class Object(gws.base.model.value.Object):
    """Format value object."""

    format: str
    """Format string."""

    def configure(self):
        self.format = self.cfg('format')

    def compute(self, field, feature, mc):
        return gws.u.format_map(self.format, feature.attributes)
