"""Feature select widget."""

import gws
import gws.base.model.widget

gws.ext.new.modelWidget('featureSelect')


class Config(gws.base.model.widget.Config):
    withSearch: bool = False


class Props(gws.base.model.widget.Props):
    withSearch: bool


class Object(gws.base.model.widget.Object):
    def props(self, user):
        return gws.u.merge(super().props(user), withSearch=self.cfg('withSearch', default=False))
