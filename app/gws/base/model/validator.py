"""Base model validator."""

import gws
import gws.types as t

DEFAULT_MESSAGE_PREFIX = 'validationError_'


class Config(gws.Config):
    message: str
    forCreate: bool = True
    forUpdate: bool = True


class Object(gws.Node, gws.IModelValidator):
    def configure(self):
        self.message = self.cfg('message', default=DEFAULT_MESSAGE_PREFIX + self.extType)

        self.modes = set()
        if self.cfg('forCreate'):
            self.modes.add(gws.ModelMode.create)
        if self.cfg('forUpdate'):
            self.modes.add(gws.ModelMode.update)
