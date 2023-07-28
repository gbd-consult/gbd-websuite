import gws
import gws.types as t


class Config(gws.Config):
    isDefault: bool = False
    forRead: bool = True
    forWrite: bool = True
    forCreate: bool = True


class Object(gws.Node, gws.IModelValue):
    def configure(self):
        self.isDefault = self.cfg('isDefault', default=False)
        self.forRead = self.cfg('forRead', default=True)
        self.forWrite = self.cfg('forWrite', default=True)
        self.forCreate = self.cfg('forCreate', default=True)
