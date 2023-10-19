import gws
import gws.types as t


class Config(gws.Config):
    isDefault: bool = False
    forRead: bool = True
    forCreate: bool = True
    forUpdate: bool = True


class Object(gws.Node, gws.IModelValue):
    def configure(self):
        self.isDefault = self.cfg('isDefault', default=False)

        self.modes = set()

        if self.cfg('forRead'):
            self.modes.add(gws.ModelMode.view)
            self.modes.add(gws.ModelMode.edit)
        if self.cfg('forCreate'):
            self.modes.add(gws.ModelMode.create)
        if self.cfg('forUpdate'):
            self.modes.add(gws.ModelMode.update)
