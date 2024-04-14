"""Textarea widget."""

import gws
import gws.base.model.widget
import gws.types as t

gws.ext.new.modelWidget('textarea')


class Config(gws.base.model.widget.Config):
    height: int = 0
    """textarea height placeholder"""
    placeholder: str = ''
    """textarea placeholder"""


class Props(gws.base.model.widget.Props):
    height: int
    placeholder: str


class Object(gws.base.model.widget.Object):

    def props(self, user):
        return gws.u.merge(super().props(user), placeholder=self.cfg('placeholder'), height=self.cfg('height'))
