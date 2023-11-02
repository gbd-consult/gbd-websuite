"""Feature select widget."""

import gws
import gws.base.model.widget

gws.ext.new.modelWidget('geometry')


class Config(gws.base.model.widget.Config):
    pass


class Props(gws.base.model.widget.Props):
    pass


class Object(gws.base.model.widget.Object):
    supportsTableView = False
