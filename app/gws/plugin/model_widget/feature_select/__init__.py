"""Feature select widget."""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('featureSelect')
class Config(gws.base.model.widget.Config):
    """Feature select widget configuration."""

    withSearch: bool = False
    """Enable search functionality in the widget."""


@gws.ext.props.modelWidget('featureSelect')
class Props(gws.base.model.widget.Props):
    withSearch: bool


@gws.ext.object.modelWidget('featureSelect')
class Object(gws.base.model.widget.Object):
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            withSearch=self.cfg('withSearch', default=False),
        )
