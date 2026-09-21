"""File widget."""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('file')
class Config(gws.base.model.widget.Config):
    """File widget configuration."""

    pass


@gws.ext.props.modelWidget('file')
class Props(gws.base.model.widget.Props):
    pass


@gws.ext.object.modelWidget('file')
class Object(gws.base.model.widget.Object):
    pass
