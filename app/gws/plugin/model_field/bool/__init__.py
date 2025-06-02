"""Boolean field."""

import gws
import gws.base.model.scalar_field

gws.ext.new.modelField('bool')


class Config(gws.base.model.scalar_field.Config):
    """Configuration for boolean field."""

    pass


class Props(gws.base.model.scalar_field.Props):
    pass


class Object(gws.base.model.scalar_field.Object):
    attributeType = gws.AttributeType.bool

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='toggle')
            return True

    ##

    def prop_to_python(self, feature, value, mc):
        try:
            return bool(value)
        except ValueError:
            return gws.ErrorValue
