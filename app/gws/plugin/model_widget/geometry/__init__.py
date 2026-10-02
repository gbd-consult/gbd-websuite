"""Feature select widget."""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('geometry')
class Config(gws.base.model.widget.Config):
    """Buttons to draw or edit the feature geometry."""

    isInline: bool = False
    """Display the geometry widget in the form."""
    withText: bool = False
    """Show a button to edit the geometry as text."""


@gws.ext.props.modelWidget('geometry')
class Props(gws.base.model.widget.Props):
    isInline: bool
    withText: bool


@gws.ext.object.modelWidget('geometry')
class Object(gws.base.model.widget.Object):
    supportsTableView = False

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            isInline=self.cfg('isInline', default=False),
            withText=self.cfg('withText', default=False),
        )
