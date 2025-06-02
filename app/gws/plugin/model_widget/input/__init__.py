"""Simple input widget."""

import gws
import gws.base.model.widget

gws.ext.new.modelWidget('input')


class Config(gws.base.model.widget.Config):
    """Generic input widget configuration."""

    placeholder: str = ''
    """Input box placeholder."""


class Props(gws.base.model.widget.Props):
    placeholder: str


class Object(gws.base.model.widget.Object):
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            placeholder=self.cfg('placeholder'),
        )
