"""Base class for field widgets."""

import gws


class Props(gws.Props):
    type: str
    readOnly: bool
    uid: str


class Config(gws.Config):
    readOnly: bool = False


class Object(gws.ModelWidget):
    readOnly: bool

    def configure(self):
        self.readOnly = self.cfg('readOnly', default=False)

    def props(self, user):
        return Props(type=self.extType, readOnly=self.readOnly, uid=self.uid)
