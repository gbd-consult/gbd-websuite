"""Base class for field widgets."""

import gws


class Props(gws.Props):
    type: str
    readOnly: bool
    uid: str


class Config(gws.Config):
    pass


class Object(gws.Node, gws.IModelWidget):

    def props(self, user):
        return Props(type=self.extType, readOnly=False, uid=self.uid)
