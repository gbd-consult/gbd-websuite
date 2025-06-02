"""Integer field."""

import gws
import gws.base.model.scalar_field
from gws import User

gws.ext.new.modelField('integer')


class Config(gws.base.model.scalar_field.Config):
    """Configuration for integer field."""

    pass


class Props(gws.base.model.scalar_field.Props):
    pass


class Object(gws.base.model.scalar_field.Object):
    attributeType = gws.AttributeType.int

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='integer')
            return True

    def prop_to_python(self, feature, value, mc):
        try:
            return int(value)
        except ValueError:
            return gws.ErrorValue
