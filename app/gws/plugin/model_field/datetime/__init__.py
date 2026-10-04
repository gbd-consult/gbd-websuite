"""Datetime field.

A scalar field for date and time values. Values from the client are parsed
with ``gws.lib.datetimex.parse`` and stored as ``datetime`` objects. They are
always sent to the client as ISO strings; locale-specific formatting is left
to the client. Without a configured widget, the field uses an ``input`` widget.

Example::

    fields+ { name "updated_at" type "datetime" title "Last update" }
"""

import gws
import gws.base.model.scalar_field
import gws.lib.datetimex


@gws.ext.config.modelField('datetime')
class Config(gws.base.model.scalar_field.Config):
    """Field for date and time values."""

    pass


@gws.ext.props.modelField('datetime')
class Props(gws.base.model.scalar_field.Props):
    pass


@gws.ext.object.modelField('datetime')
class Object(gws.base.model.scalar_field.Object):
    """Datetime field object."""

    attributeType = gws.AttributeType.datetime

    def configure_widget(self):
        if not super().configure_widget():
            # @TODO datetime widget
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='input')
            return True

    def prop_to_python(self, feature, value, mc):
        d = gws.lib.datetimex.parse(value)
        return d or gws.ErrorValue

    def python_to_prop(self, feature, value, mc):
        return gws.lib.datetimex.to_iso_string(value)
