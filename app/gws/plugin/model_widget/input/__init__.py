"""Simple input widget."""

import gws
import gws.base.model.widget

gws.ext.new.modelWidget('input')


class Config(gws.base.model.widget.Config):
    placeholder: str = ''
    """input box placeholder"""


class Props(gws.base.model.widget.Props):
    placeholder: str


class Object(gws.base.model.widget.Object):
    def props(self, user):
        return gws.u.merge(super().props(user), placeholder=self.cfg('placeholder'))
