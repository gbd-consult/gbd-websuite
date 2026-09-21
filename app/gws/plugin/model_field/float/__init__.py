"""Integer field."""

import gws
import gws.base.model.scalar_field


@gws.ext.config.modelField('float')
class Config(gws.base.model.scalar_field.Config):
    """Configuration for float field."""

    pass


@gws.ext.props.modelField('float')
class Props(gws.base.model.scalar_field.Props):
    pass


@gws.ext.object.modelField('float')
class Object(gws.base.model.scalar_field.Object):
    """Float field object."""

    attributeType = gws.AttributeType.float

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='float')
            return True

    def prop_to_python(self, feature, value, mc):
        try:
            return float(value)
        except ValueError:
            return gws.ErrorValue
