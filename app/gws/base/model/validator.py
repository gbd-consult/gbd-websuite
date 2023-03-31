import gws
import gws.types as t


class Config(gws.Config):
    message: str
    forWrite: bool = True
    forCreate: bool = True


class Object(gws.Node, gws.IModelValidator):
    def configure(self):
        self.message = self.cfg('message')
        self.forWrite = self.cfg('forWrite', default=True)
        self.forCreate = self.cfg('forCreate', default=True)
