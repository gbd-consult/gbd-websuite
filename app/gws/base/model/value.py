"""Base model value."""

import gws


class Config(gws.Config):
    """Configuration for the model value."""

    isDefault: bool = False
    """The value is used only when no other value is provided."""
    forRead: bool = True
    """The value is applied when reading an object."""
    forCreate: bool = True
    """The value is applied when creating a new object."""
    forUpdate: bool = True
    """The value is applied when updating an existing object."""


class Object(gws.ModelValue):
    """Base model value.

    Provides the ``isDefault`` flag and the set of operations the value applies to.
    Subclasses implement ``compute``.
    """

    def configure(self):
        self.isDefault = self.cfg('isDefault', default=False)

        self.ops = set()

        if self.cfg('forRead'):
            self.ops.add(gws.ModelOperation.read)
        if self.cfg('forCreate'):
            self.ops.add(gws.ModelOperation.create)
        if self.cfg('forUpdate'):
            self.ops.add(gws.ModelOperation.update)
