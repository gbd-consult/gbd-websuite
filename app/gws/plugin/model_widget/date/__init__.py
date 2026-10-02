"""Date widget."""

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
    pass
