"""Date field.

Internally, ``date`` values are ``datetime`` objects.
They are always transferred in the ISO format,
specific locale conversions are left to the client.
"""

import gws
import gws.base.model.field
import gws.lib.date

gws.ext.new.modelField('date')


class Config(gws.base.model.field.Config):
    pass


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Scalar):
    attributeType = gws.AttributeType.date

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, {'type': 'date'})
            return True

    def prop_to_py(self, val):
        d = gws.lib.date.parse(val)
        return d.date() if d else gws.ErrorValue

    def py_to_db(self, val):
        return gws.lib.date.to_iso_date_string(val)

    def py_to_prop(self, val):
        return gws.lib.date.to_iso_date_string(val)
