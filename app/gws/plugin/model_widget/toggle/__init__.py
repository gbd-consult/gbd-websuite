"""Toggle input widget.

Checkbox or radio button for boolean values. This is the default widget of
``bool`` fields.

Example::

    fields+ {
        name "is_active"
        type "bool"
        widget { type "toggle" kind "checkbox" }
    }
"""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('toggle')
class Config(gws.base.model.widget.Config):
    """Checkbox or radio button for boolean values."""

    kind: str = 'checkbox'
    """Toggle kind: checkbox or radio."""


@gws.ext.props.modelWidget('toggle')
class Props(gws.base.model.widget.Props):
    kind: str


@gws.ext.object.modelWidget('toggle')
class Object(gws.base.model.widget.Object):
    """Toggle input widget object."""

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            kind=self.cfg('kind', default='checkbox'),
        )
