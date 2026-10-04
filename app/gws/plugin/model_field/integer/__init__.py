"""Integer field.

A scalar field for integer numbers. Values from the client are converted
with ``int()``. Without a configured widget, the field uses an ``integer``
widget.

Example::

    fields+ { name "id" type "integer" isPrimaryKey true }
"""

import gws
import gws.base.model.scalar_field
from gws import User


@gws.ext.config.modelField('integer')
class Config(gws.base.model.scalar_field.Config):
    """Field for integer numbers."""

    pass


@gws.ext.props.modelField('integer')
class Props(gws.base.model.scalar_field.Props):
    pass


@gws.ext.object.modelField('integer')
class Object(gws.base.model.scalar_field.Object):
    """Integer field object."""

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
