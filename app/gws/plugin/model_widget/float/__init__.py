"""Float input widget."""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('float')
class Config(gws.base.model.widget.Config):
    """Float input widget configuration."""

    step: int = 1
    """Numeric step."""
    placeholder: str = ''
    """Input box placeholder."""


@gws.ext.props.modelWidget('float')
class Props(gws.base.model.widget.Props):
    step: int
    placeholder: str


@gws.ext.object.modelWidget('float')
class Object(gws.base.model.widget.Object):
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            placeholder=self.cfg('placeholder'),
            step=self.cfg('step'),
        )
