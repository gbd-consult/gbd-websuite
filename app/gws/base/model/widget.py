"""Base class for field widgets."""

import gws


class Props(gws.Props):
    type: str
    readOnly: bool
    uid: str


class Config(gws.Config):
    """Configuration for the widget."""

    readOnly: bool = False
    """If True, the value is read-only."""


class Object(gws.ModelWidget):
    readOnly: bool

    def configure(self):
        self.readOnly = self.cfg('readOnly', default=False)

    def props(self, user):
        return Props(type=self.extType, readOnly=self.readOnly, uid=self.uid)
