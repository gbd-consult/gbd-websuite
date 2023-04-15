"""Generic scalar field."""

import gws
import gws.lib.sa as sa

from . import field


class Config(field.Config):
    pass


class Props(field.Props):
    pass


class Object(field.Object):
    def columns(self):
        kwargs = {}
        if self.isPrimaryKey:
            kwargs['primary_key'] = True
        # if self.value.serverDefault:
        #     kwargs['server_default'] = sa.text(self.value.serverDefault)
        col = sa.Column(self.name, _SCALAR_TYPES[self.attributeType], **kwargs)
        return [col]


_SCALAR_TYPES = {
    gws.AttributeType.bool: sa.Boolean,
    gws.AttributeType.date: sa.Date,
    gws.AttributeType.datetime: sa.DateTime,
    gws.AttributeType.float: sa.Float,
    gws.AttributeType.int: sa.Integer,
    gws.AttributeType.str: sa.String,
    gws.AttributeType.time: sa.Time,
    gws.AttributeType.geometry: sa.geo.Geometry,
}
