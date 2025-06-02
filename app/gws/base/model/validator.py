"""Base model validator."""

import gws

DEFAULT_MESSAGE_PREFIX = 'validationError_'


class Config(gws.Config):
    """Configuration for the model validator."""

    message: str = ''
    """Error message prefix for validation errors."""
    forCreate: bool = True
    """If True, the validator is applied when creating a new object."""
    forUpdate: bool = True
    """If True, the validator is applied when updating an existing object."""


class Object(gws.ModelValidator):
    def configure(self):
        self.message = self.cfg('message') or DEFAULT_MESSAGE_PREFIX + self.extType

        self.ops = set()
        if self.cfg('forCreate'):
            self.ops.add(gws.ModelOperation.create)
        if self.cfg('forUpdate'):
            self.ops.add(gws.ModelOperation.update)
