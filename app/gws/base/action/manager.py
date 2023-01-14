import gws
import gws.spec
import gws.types as t
import gws.base.web.error


class Props(gws.Props):
    type: str


class Config(gws.ConfigWithAccess):
    """Server actions"""

    actions: t.Optional[t.List[gws.ext.config.action]] 
    """available actions"""


class Object(gws.Node, gws.IActionManager):
    index: t.Dict[str, gws.IAction]

    def configure(self):
        self.items = self.create_children(gws.ext.object.action, self.var('actions'))
        self.index = {}
        for a in self.items:
            self.index[a.extName] = a
            self.index[a.extType] = a
            self.index[gws.class_name(a)] = a

    def actions_for(self, user, other=None):
        d = {}
        for a in self.items:
            if user.can_use(a):
                d[a.extType] = a
        if other:
            for a in other.items:
                if user.can_use(a):
                    d[a.extType] = a
        return list(d.values())

    def get_action(self, desc):
        return self.index.get(desc.owner.extName)
