"""Integer input widget.

Input for whole numbers, with up/down buttons and an optional placeholder
text. This is the default widget of ``integer`` fields.

Example::

    fields+ {
        name "floors"
        type "integer"
        widget { type "integer" step 1 placeholder "Number of floors" }
    }
"""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('integer')
class Config(gws.base.model.widget.Config):
    """Input for whole numbers."""

    step: int = 1
    """Increment of the up/down buttons."""
    placeholder: str = ''
    """Hint text shown in the empty input."""


@gws.ext.props.modelWidget('integer')
class Props(gws.base.model.widget.Props):
    step: int
    placeholder: str


@gws.ext.object.modelWidget('integer')
class Object(gws.base.model.widget.Object):
    """Integer input widget object."""

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            placeholder=self.cfg('placeholder'),
            step=self.cfg('step'),
        )
