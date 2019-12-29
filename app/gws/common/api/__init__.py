import gws
import gws.types as t


class Config(t.Config):
    """Server actions"""

    access: t.Optional[t.Access]  #: default access mode
    actions: t.Optional[t.List[t.ext.action.Config]]  #: available actions


#:stub ApiObject
class Object(gws.Object):
    def __init__(self):
        super().__init__()
        self.actions = {}

    def configure(self):
        super().configure()

        for p in self.var('actions', []):
            a = self.add_child('gws.ext.action', p)
            self.actions[p.type] = a
