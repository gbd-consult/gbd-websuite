"""Boolean field."""

import gws
import gws.base.model.scalar_field


@gws.ext.config.modelField('bool')
class Config(gws.base.model.scalar_field.Config):
    """Field for boolean values."""

    pass


@gws.ext.props.modelField('bool')
class Props(gws.base.model.scalar_field.Props):
    pass


@gws.ext.object.modelField('bool')
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
