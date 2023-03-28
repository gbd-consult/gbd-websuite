"""Integer field."""

import gws
import gws.base.database.model
import gws.base.model.fields.scalar
import gws.types as t

gws.ext.new.modelField('float')


class Config(gws.base.model.fields.scalar.Config):
    pass


class Props(gws.base.model.fields.scalar.Props):
    pass


class Object(gws.base.model.fields.scalar.Object):
    """Float field object."""

    attributeType = gws.AttributeType.float

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, {'type': 'integer'})
            return True

    def convert_load(self, value):
        try:
            return float(value)
        except ValueError:
            return gws.ErrorValue
