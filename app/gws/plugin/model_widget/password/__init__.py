"""Password widget."""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('password')
class Config(gws.base.model.widget.Config):
    """Password widget."""

    placeholder: str = ''
    """Password input placeholder."""
    withShow: bool = False
    """Create a "show password" button."""


@gws.ext.props.modelWidget('password')
class Props(gws.base.model.widget.Props):
    placeholder: str
    withShow: bool


@gws.ext.object.modelWidget('password')
class Object(gws.base.model.widget.Object):
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            placeholder=self.cfg('placeholder'),
            withShow=self.cfg('withShow'),
        )
