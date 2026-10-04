"""Float input widget.

Input for decimal numbers, with up/down buttons and an optional placeholder
text. This is the default widget of ``float`` fields.

Example::

    fields+ {
        name "area"
        type "float"
        widget { type "float" step 10 placeholder "Area in m2" }
    }
"""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('float')
class Config(gws.base.model.widget.Config):
    """Input for decimal numbers."""

    step: int = 1
    """Increment of the up/down buttons."""
    placeholder: str = ''
    """Hint text shown in the empty input."""


@gws.ext.props.modelWidget('float')
class Props(gws.base.model.widget.Props):
    step: int
    placeholder: str


@gws.ext.object.modelWidget('float')
class Object(gws.base.model.widget.Object):
    """Float input widget object."""

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            placeholder=self.cfg('placeholder'),
            step=self.cfg('step'),
        )
