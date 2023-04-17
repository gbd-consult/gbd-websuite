"""Textarea widget."""

import gws
import gws.base.model.widget
import gws.types as t

gws.ext.new.modelWidget('textarea')


class Config(gws.base.model.widget.Config):
    height: int = 0


class Props(gws.base.model.widget.Props):
    height: int


class Object(gws.base.model.widget.Object):

    def props(self, user):
        return gws.merge(super().props(user), height=self.cfg('height'))
