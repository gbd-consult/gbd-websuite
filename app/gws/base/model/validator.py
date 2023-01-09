import gws
import gws.types as t


class Config(gws.Config):
    message: str
    forWrite: bool = True
    forCreate: bool = True


class Object(gws.Node, gws.IModelValidator):
    def configure(self):
        self.message = self.var('message')
        self.forWrite = self.var('forWrite', default=True)
        self.forCreate = self.var('forCreate', default=True)
