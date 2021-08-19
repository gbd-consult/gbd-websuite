import gws
import gws.types as t
from . import action


class Config(gws.WithAccess):
    """Server actions"""

    actions: t.Optional[t.List[gws.ext.action.Config]]  #: available actions


class Object(gws.Object):
    actions: t.Dict

    def configure(self):
        self.actions = {}

        for p in self.var('actions', []):
            a = self.create_child('gws.ext.action', p)
            if a:
                self.actions[a.ext_type] = a

    def find_action(self, action_type: str):
        return self.actions.get(action_type)
