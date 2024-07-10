"""Password widget."""

import gws
import gws.base.model.widget

gws.ext.new.modelWidget('password')


class Config(gws.base.model.widget.Config):
    placeholder: str = ''
    """Password input placeholder."""
    withShow: bool = False
    """Create a "show password" button."""


class Props(gws.base.model.widget.Props):
    placeholder: str
    withShow: bool


class Object(gws.base.model.widget.Object):
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            placeholder=self.cfg('placeholder'),
            withShow=self.cfg('withShow'),
        )
