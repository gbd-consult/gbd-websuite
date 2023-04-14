"""Float input widget."""

import gws
import gws.base.model.widget

gws.ext.new.modelWidget('float')


class Config(gws.base.model.widget.Config):
    step: int = 1


class Props(gws.base.model.widget.Props):
    step: int


class Object(gws.base.model.widget.Object):

    def props(self, user):
        return gws.merge(super().props(user), step=self.cfg('step'))
