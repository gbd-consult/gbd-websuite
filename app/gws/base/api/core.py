import gws
import gws.types as t

from . import action


class Config(gws.ConfigWithAccess):
    """Server actions"""

    actions: t.Optional[t.List[gws.ext.config.action]]  #: available actions


# @TODO allow multiple actions with the same ext_type with different permissions

class Object(gws.Node, gws.IApi):
    actions: t.List[gws.Node]

    def configure(self):
        self.actions = self.create_children('gws.ext.action', self.var('actions'))

    def actions_for(self, user, parent=None):
        ds = parent.actions_for(user) if parent else {}
        for a in self.actions:
            ds[a.ext_type] = a
        return ds


class Error(gws.Error):
    pass

class CommandNotFound(Error):
    pass

class CommandNotAllowed(Error):
    pass


