"""Simple checkbox input widget."""

import gws
import gws.base.model.widget
import gws.types as t

gws.ext.new.modelWidget('checkbox')


class Config(gws.base.model.widget.Config):
    pass

class Props(gws.base.model.widget.Props):
    pass

class Object(gws.base.model.widget.Object):

    def props(self, user):
        #return super().props(user)
        return gws.merge(super().props(user))
