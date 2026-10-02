"""Simple input widget."""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('input')
class Config(gws.base.model.widget.Config):
    """Single-line text input."""

    placeholder: str = ''
    """Hint text shown in the empty input."""


@gws.ext.props.modelWidget('input')
class Props(gws.base.model.widget.Props):
    placeholder: str


@gws.ext.object.modelWidget('input')
class Object(gws.base.model.widget.Object):
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            placeholder=self.cfg('placeholder'),
        )
