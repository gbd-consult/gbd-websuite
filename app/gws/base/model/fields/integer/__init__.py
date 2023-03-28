"""Integer field."""

import gws
import gws.base.database.model
import gws.base.model.fields.scalar
import gws.types as t

gws.ext.new.modelField('integer')


class Config(gws.base.model.fields.scalar.Config):
    pass


class Props(gws.base.model.fields.scalar.Props):
    pass


class Object(gws.base.model.fields.scalar.Object):
    """Integer field object."""

    attributeType = gws.AttributeType.int

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, {'type': 'integer'})
            return True

    def convert_load(self, value):
        try:
            return int(value)
        except ValueError:
            return gws.ErrorValue
