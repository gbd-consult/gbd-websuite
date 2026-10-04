"""Password widget.

Text input that masks the typed characters, with an optional placeholder text
and an optional button that reveals the password.

Example::

    fields+ {
        name "password"
        type "text"
        widget { type "password" withShow true }
    }
"""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('password')
class Config(gws.base.model.widget.Config):
    """Password input with masked characters."""

    placeholder: str = ''
    """Hint text shown in the empty input."""
    withShow: bool = False
    """Show a button that reveals the typed password."""


@gws.ext.props.modelWidget('password')
class Props(gws.base.model.widget.Props):
    placeholder: str
    withShow: bool


@gws.ext.object.modelWidget('password')
class Object(gws.base.model.widget.Object):
    """Password widget object."""

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            placeholder=self.cfg('placeholder'),
            withShow=self.cfg('withShow'),
        )
