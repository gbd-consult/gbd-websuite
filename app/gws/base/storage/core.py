import gws
import gws.base.web.error
import gws.lib.json2
import gws.types as t

from . import types
from .providers import sqlite

# @TODO: more provider types
# @TODO: granular role-based permissions
# @TODO: direct API (.read, .write etc)


@gws.ext.config.helper('storage')
class Config(gws.Config):
    """Storage helper"""

    path: t.Optional[str]  #: path to the storage file
    permissions: t.Optional[t.List[types.PermissionRule]]  #: permission rules


class Permission(gws.Node):
    category: str
    mode: types.PermissionMode

    def configure(self):
        self.category = self.var('category')
        self.mode = self.var('mode')




@gws.ext.object.helper('storage')
class Object(gws.Node):
    provider: sqlite.Object
    permissions: t.List[Permission]

    def configure(self):
        self.permissions = self.create_children(Permission, self.var('permissions'))
        self.provider = self.require_child(sqlite.Object, self.config)

    def handle_action(self, req: gws.IWebRequest, p: types.Params, category: str) -> types.Response:
        readable, writable = self._category_permissions(category, req.user)

        data = None
        entries = []

        if p.verb == types.Verb.list:
            pass

        if p.verb == types.Verb.read:
            if not readable or not p.entryName:
                raise gws.base.web.error.Forbidden()
            rec = self.provider.read(category, p.entryName)
            if rec:
                data = gws.lib.json2.from_string(rec.data)

        if p.verb == types.Verb.write:
            if not writable or not p.entryName or not p.entryData:
                raise gws.base.web.error.Forbidden()
            d = gws.lib.json2.to_string(p.entryData)
            self.provider.write(category, p.entryName, d, req.user.uid)

        if p.verb == types.Verb.delete:
            if not writable or not p.entryName:
                raise gws.base.web.error.Forbidden()
            self.provider.delete(category, p.entryName)

        if readable:
            entries = [types.Entry(name=r.name) for r in self.provider.list(category)]

        return types.Response(
            data=data,
            directory=types.Directory(
                category=category,
                readable=readable,
                writable=writable,
                entries=entries
            ))

    def _category_permissions(self, category: str, user: gws.IUser):
        r = False
        w = False

        for p in self.permissions:
            if (p.category == category or p.category == '*') and user.can_use(p):
                if p.mode == types.PermissionMode.all:
                    r = w = True
                elif p.mode == types.PermissionMode.read:
                    r = True
                elif p.mode == types.PermissionMode.write:
                    w = True

        return r, w
