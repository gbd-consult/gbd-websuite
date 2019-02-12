import gws
import gws.types as t


class Config(t.Config):
    """Server actions"""

    access: t.Optional[t.Access]  #: default access mode
    actions: t.Optional[t.List[t.ext.action.Config]]  #: available actions


class Object(gws.PublicObject):
    def configure(self):
        super().configure()

        self.actions = {}

        for p in self.var('actions', []):
            a = self.add_child('gws.ext.action', p)
            self.actions[p.type] = a
