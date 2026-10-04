"""Date field.

A scalar field for dates. Values from the client are parsed with
``gws.lib.datetimex.parse`` and stored as ``datetime.date`` objects.
They are always sent to the client as ISO date strings (``YYYY-MM-DD``);
locale-specific formatting is left to the client. Without a configured
widget, the field uses a ``date`` widget.

Example::

    fields+ { name "start_date" type "date" title "Start date" }
"""

import gws
import gws.base.model.scalar_field
import gws.lib.datetimex


@gws.ext.config.modelField('date')
class Config(gws.base.model.scalar_field.Config):
    """Field for date values."""
    
    pass


@gws.ext.props.modelField('date')
class Props(gws.base.model.scalar_field.Props):
    pass


@gws.ext.object.modelField('date')
class Object(gws.base.model.scalar_field.Object):
    """Date field object."""

    attributeType = gws.AttributeType.date

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='date')
            return True

    def prop_to_python(self, feature, value, mc):
        d = gws.lib.datetimex.parse(value)
        return gws.lib.datetimex.dt.date(d.year, d.month, d.day) if d else gws.ErrorValue

    def python_to_prop(self, feature, value, mc):
        return gws.lib.datetimex.to_iso_date_string(value)
