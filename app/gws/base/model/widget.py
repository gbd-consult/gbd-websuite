"""Base class for field widgets."""

import gws


class Props(gws.Props):
    type: str
    options: dict
    readOnly: bool


class Config(gws.Config):
    pass


class Object(gws.Node, gws.IModelWidget):

    def props(self, user):
        return Props(
            type=self.extType,
            options={},
            readOnly=False,
        )
