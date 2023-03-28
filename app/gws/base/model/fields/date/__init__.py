"""Date field.

Internally, ``date`` values are ``datetime`` objects.
They are always transferred in the ISO format,
specific locale conversions are left to the client.
"""

import gws
import gws.lib.date

from .. import scalar

gws.ext.new.modelField('date')


class Config(scalar.Config):
    pass


class Props(scalar.Props):
    pass


class Object(scalar.Object):
    """Date field object."""

    attributeType = gws.AttributeType.int

    def convert_load(self, value):
        try:
            return gws.lib.date.parse(value).date()
        except Exception:
            return gws.ErrorValue

    def convert_store(self, value):
        return gws.lib.date.to_iso_date(value)
