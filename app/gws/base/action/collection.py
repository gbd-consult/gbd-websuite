import gws
import gws.spec
import gws.types as t
import gws.base.web.error


class Props(gws.Props):
    type: str


class Config(gws.ConfigWithAccess):
    """Server actions"""

    actions: t.Optional[t.List[gws.ext.config.action]]  #: available actions


class Object(gws.Node):
    index: t.Dict[str, gws.IAction]
    actions: t.List[gws.IAction]

    def configure(self):
        self.actions = self.create_children(gws.ext.object.action, self.var('actions'))
        self.index = {}
        for a in self.actions:
            self.index[a.extName] = a
            self.index[a.extType] = a
            self.index[gws.class_name(a)] = a

    def find(self, class_name):
        return self.index.get(class_name)

