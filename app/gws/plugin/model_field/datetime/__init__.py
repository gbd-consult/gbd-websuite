"""Datetime field.

Internally, ``datetime`` values are ``datetime`` objects.
They are always transferred in the ISO format,
specific locale conversions are left to the client.
"""

import gws
import gws.base.model.scalar_field
import gws.lib.datetimex

gws.ext.new.modelField('datetime')


class Config(gws.base.model.scalar_field.Config):
    pass


class Props(gws.base.model.scalar_field.Props):
    pass


class Object(gws.base.model.scalar_field.Object):
    attributeType = gws.AttributeType.datetime

    def configure_widget(self):
        if not super().configure_widget():
            # @TODO datetime widget
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='input')
            return True

    def prop_to_python(self, feature, value, mc):
        d = gws.lib.datetimex.parse(value)
        return d or gws.ErrorValue

    def python_to_prop(self, feature, value, mc):
        return gws.lib.datetimex.to_iso_string(value)
