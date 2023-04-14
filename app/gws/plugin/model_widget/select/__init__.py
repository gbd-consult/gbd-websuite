"""Select widget."""

import gws
import gws.base.model.widget
import gws.types as t

gws.ext.new.modelWidget('select')


class Config(gws.base.model.widget.Config):
    items: list[t.Any]


class Props(gws.base.model.widget.Props):
    items: list[t.Any]


class Object(gws.base.model.widget.Object):

    def props(self, user):
        return gws.merge(super().props(user), items=self.cfg('items'))
