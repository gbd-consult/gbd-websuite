"""Simple input widget."""

import gws
import gws.base.model.widget
import gws.types as t


@gws.ext.props.modelWidget('input')
class Props(gws.base.model.widget.Props):
    pass


@gws.ext.config.modelWidget('input')
class Config(gws.base.model.widget.Config):
    pass


@gws.ext.object.modelWidget('input')
class Object(gws.base.model.widget.Object):
    pass
