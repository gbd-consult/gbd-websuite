"""Simple input widget.

Single-line text input with an optional placeholder text. This is the default
widget of ``text``, ``datetime`` and ``time`` fields.

Example::

    fields+ {
        name "name"
        type "text"
        widget { type "input" placeholder "Name" }
    }
"""

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
    """Simple input widget object."""

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            placeholder=self.cfg('placeholder'),
        )
