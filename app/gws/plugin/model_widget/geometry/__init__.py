"""Feature select widget."""

import gws
import gws.base.model.widget

gws.ext.new.modelWidget('geometry')


class Config(gws.base.model.widget.Config):
    """Geometry widget configuration."""

    isInline: bool = False
    """Display the geometry widget in the form."""
    withText: bool = False
    """Display the text geometry editor."""


class Props(gws.base.model.widget.Props):
    isInline: bool
    withText: bool


class Object(gws.base.model.widget.Object):
    supportsTableView = False

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            isInline=self.cfg('isInline', default=False),
            withText=self.cfg('withText', default=False),
        )
