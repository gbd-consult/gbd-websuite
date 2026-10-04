"""Textarea widget.

Multi-line text input with an optional height and placeholder text.

Example::

    fields+ {
        name "description"
        type "text"
        widget { type "textarea" height 120 }
    }
"""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('textarea')
class Config(gws.base.model.widget.Config):
    """Multi-line text input."""

    height: int = 0
    """Height of the input in pixels."""
    placeholder: str = ''
    """Hint text shown in the empty input."""


@gws.ext.props.modelWidget('textarea')
class Props(gws.base.model.widget.Props):
    height: int
    placeholder: str


@gws.ext.object.modelWidget('textarea')
class Object(gws.base.model.widget.Object):
    """Textarea widget object."""

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            placeholder=self.cfg('placeholder'),
            height=self.cfg('height'),
        )
