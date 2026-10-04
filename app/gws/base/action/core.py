"""Base action object."""

import gws


class Props(gws.Props):
    type: str


class Config(gws.ConfigWithAccess):
    pass


class Object(gws.Action):
    """Base action object, the parent of all action objects.

    An action groups server commands, which subclasses implement as methods
    decorated with ``gws.ext.command.*``. The base class provides the action
    props, which contain only the action type.
    """

    def props(self, user):
        return gws.Data(type=self.extType)
