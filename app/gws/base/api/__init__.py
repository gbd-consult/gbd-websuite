import gws
import gws.types as t
from . import action


class Config(gws.WithAccess):
    """Server actions"""

    actions: t.Optional[t.List[gws.ext.action.Config]]  #: available actions


# @TODO allow multiple actions with the same ext_type with different permissions

class Object(gws.Object, gws.IApi):
    actions: t.List[gws.Object]

    def configure(self):
        self.actions = self.create_children('gws.ext.action', self.var('actions'))

    def find_action(self, ext_type):
        for a in self.actions:
            if a.ext_type == ext_type:
                return a

    def get_actions(self, other=None):
        ls = list(self.actions)
        if other:
            ks = set(a.ext_type for a in self.actions)
            for a in other.get_actions():
                if a.ext_type not in ks:
                    ls.append(a)
        return sorted(ls, key=lambda a: a.ext_type)
