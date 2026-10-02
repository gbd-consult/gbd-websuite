"""Toggle input widget."""

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
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            kind=self.cfg('kind', default='checkbox'),
        )
