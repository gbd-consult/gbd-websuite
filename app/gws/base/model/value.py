import gws
import gws.types as t


class Config(gws.Config):
    isDefault: bool
    forRead: bool = True
    forWrite: bool = True
    forCreate: bool = True


class Object(gws.Node, gws.IModelValue):
    def configure(self):
        self.isDefault = self.var('isDefault')
        self.forRead = self.var('forRead', default=True)
        self.forWrite = self.var('forWrite', default=True)
        self.forCreate = self.var('forCreate', default=True)
