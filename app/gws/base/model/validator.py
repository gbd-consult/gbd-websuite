"""Base model validator."""

import gws
import gws.types as t

DEFAULT_MESSAGE_PREFIX = 'validationError_'


class Config(gws.Config):
    message: str
    forCreate: bool = True
    forUpdate: bool = True


class Object(gws.ModelValidator):
    def configure(self):
        self.message = self.cfg('message', default=DEFAULT_MESSAGE_PREFIX + self.extType)

        self.ops = set()
        if self.cfg('forCreate'):
            self.ops.add(gws.ModelOperation.create)
        if self.cfg('forUpdate'):
            self.ops.add(gws.ModelOperation.update)
