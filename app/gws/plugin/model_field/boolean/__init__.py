"""Boolean field."""

import gws
import gws.base.model.scalar_field

gws.ext.new.modelField('boolean')


class Config(gws.base.model.scalar_field.Config):
    pass


class Props(gws.base.model.scalar_field.Props):
    pass


class Object(gws.base.model.scalar_field.Object):
    attributeType = gws.AttributeType.bool

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, type='checkbox')
            return True

    ##

    def prop_to_py(self, val):
        try:
            return bool(val)
        except ValueError:
            return gws.ErrorValue
