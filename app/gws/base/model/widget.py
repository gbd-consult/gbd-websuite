"""Base class for field widgets."""

import gws


class Props(gws.Props):
    type: str
    readOnly: bool
    uid: str


class Config(gws.Config):
    """Configuration for the widget."""

    readOnly: bool = False
    """The widget displays the value without allowing edits."""


class Object(gws.ModelWidget):
    """Base model widget.

    Provides the ``readOnly`` flag and the props common to all widgets.
    """

    readOnly: bool
    """The widget displays the value without allowing edits."""

    def configure(self):
        self.readOnly = self.cfg('readOnly', default=False)

    def props(self, user):
        return Props(type=self.extType, readOnly=self.readOnly, uid=self.uid)
