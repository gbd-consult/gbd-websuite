"""Date widget.

Input for date values. This is the default widget of ``date`` fields.

Example::

    fields+ { name "start_date" type "date" widget.type "date" }
"""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('date')
class Config(gws.base.model.widget.Config):
    """Input for date values."""

    pass


@gws.ext.props.modelWidget('date')
class Props(gws.base.model.widget.Props):
    pass


@gws.ext.object.modelWidget('date')
class Object(gws.base.model.widget.Object):
    """Date widget object."""

    pass
