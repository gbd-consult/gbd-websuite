import gws


class Config(gws.Config):
    """Configuration for the model value."""

    isDefault: bool = False
    """If True, this value is the default for the model."""
    forRead: bool = True
    """The value is applied when reading an object."""
    forCreate: bool = True
    """The value is applied when creating a new object."""
    forUpdate: bool = True
    """The value is applied when updating an existing object."""


class Object(gws.ModelValue):
    def configure(self):
        self.isDefault = self.cfg('isDefault', default=False)

        self.ops = set()

        if self.cfg('forRead'):
            self.ops.add(gws.ModelOperation.read)
        if self.cfg('forCreate'):
            self.ops.add(gws.ModelOperation.create)
        if self.cfg('forUpdate'):
            self.ops.add(gws.ModelOperation.update)
