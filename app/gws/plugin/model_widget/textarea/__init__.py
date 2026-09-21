"""Textarea widget."""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('textarea')
class Config(gws.base.model.widget.Config):
    """Textarea widget configuration."""

    height: int = 0
    """Textarea height."""
    placeholder: str = ''
    """Textarea placeholder."""


@gws.ext.props.modelWidget('textarea')
class Props(gws.base.model.widget.Props):
    height: int
    placeholder: str


@gws.ext.object.modelWidget('textarea')
class Object(gws.base.model.widget.Object):
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            placeholder=self.cfg('placeholder'),
            height=self.cfg('height'),
        )
