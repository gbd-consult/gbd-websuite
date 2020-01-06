import gws
import gws.types as t


class Config(t.WithAccess):
    """Server actions"""

    actions: t.Optional[t.List[t.ext.action.Config]]  #: available actions


#:export IApi
class Object(gws.Object, t.IApi):
    def __init__(self):
        super().__init__()
        self.actions = {}

    def configure(self):
        super().configure()

        for p in self.var('actions', []):
            a = self.add_child('gws.ext.action', p)
            self.actions[p.type] = a
