"""Base model validator."""

import gws

DEFAULT_MESSAGE_PREFIX = 'validationError_'


class Config(gws.Config):
    message: str
    forWrite: bool = True
    forCreate: bool = True


class Object(gws.Node, gws.IModelValidator):
    def configure(self):
        self.message = self.cfg('message', default=DEFAULT_MESSAGE_PREFIX + self.extType)
        self.forWrite = self.cfg('forWrite', default=True)
        self.forCreate = self.cfg('forCreate', default=True)
