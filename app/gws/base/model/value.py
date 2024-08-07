import gws


class Config(gws.Config):
    isDefault: bool = False
    forRead: bool = True
    forCreate: bool = True
    forUpdate: bool = True


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
