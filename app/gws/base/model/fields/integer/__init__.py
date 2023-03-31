"""Integer field."""

import gws
import gws.base.model.field

gws.ext.new.modelField('integer')


class Config(gws.base.model.field.Config):
    pass


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Scalar):
    attributeType = gws.AttributeType.int

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, {'type': 'integer'})
            return True

    ##

    def prop_to_py(self, val):
        try:
            return int(val)
        except ValueError:
            return gws.ErrorValue
