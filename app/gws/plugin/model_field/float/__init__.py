"""Integer field."""

import gws
import gws.base.model.scalar_field

gws.ext.new.modelField('float')


class Config(gws.base.model.scalar_field.Config):
    pass


class Props(gws.base.model.scalar_field.Props):
    pass


class Object(gws.base.model.scalar_field.Object):
    """Float field object."""

    attributeType = gws.AttributeType.float

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, type='float')
            return True

    def prop_to_py(self, val):
        try:
            return float(val)
        except ValueError:
            return gws.ErrorValue
