"""Time field.

A scalar field for time values. Values from the client are parsed with
``gws.lib.datetimex.parse_time`` and stored as ``datetime.time`` objects.
They are always sent to the client as ISO strings; locale-specific
formatting is left to the client. Without a configured widget, the field
uses an ``input`` widget.

Example::

    fields+ { name "opens_at" type "time" title "Opening time" }
"""

import gws
import gws.base.model.scalar_field
import gws.lib.datetimex


@gws.ext.config.modelField('time')
class Config(gws.base.model.scalar_field.Config):
    """Field for time values."""

    pass


@gws.ext.props.modelField('time')
class Props(gws.base.model.scalar_field.Props):
    pass


@gws.ext.object.modelField('time')
class Object(gws.base.model.scalar_field.Object):
    """Time field object."""

    attributeType = gws.AttributeType.time

    def configure_widget(self):
        if not super().configure_widget():
            # @TODO time widget
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='input')
            return True

    def prop_to_python(self, feature, value, mc):
        d = gws.lib.datetimex.parse_time(value)
        return d if d else gws.ErrorValue

    def python_to_prop(self, feature, value, mc):
        return gws.lib.datetimex.time_to_iso_string(value)
