import gws
import gws.base.web.error
import gws.lib.json2
import gws.types as t

from . import types
from .providers import sqlite

# @TODO: more provider types
# @TODO: granular role-based permissions
# @TODO: direct API (.read, .write etc)


class PermissionMode(t.Enum):
    read = 'read'  #: an object can be read
    write = 'write'  #: an object can be written and deleted
    all = 'all'  #: an object can be read and written


class PermissionRule(gws.ConfigWithAccess):
    """Permission rule for a storage category"""

    category: str  #: storage category name
    mode: PermissionMode  #: allowed mode (read/write)


# @TODO more props, like author, time etc

class Entry(gws.Data):
    name: str


class Directory(gws.Data):
    category: str
    writable: bool
    readable: bool
    entries: t.List[Entry]


class Record(gws.Data):
    category: str
    name: str
    user_uid: str
    data: str
    created: int
    updated: int


class Verb(t.Enum):
    read = 'read'
    write = 'write'
    list = 'list'
    delete = 'delete'


class Params(gws.Request):
    verb: Verb
    entryName: t.Optional[str]
    entryData: t.Optional[dict]


class Response(gws.Response):
    directory: Directory
    data: dict

@gws.ext.config.helper('storage')
class Config(gws.Config):
    """Storage helper"""

    path: t.Optional[str]  #: path to the storage file
    permissions: t.Optional[t.List[PermissionRule]]  #: permission rules


class Permission(gws.Node):
    category: str
    mode: PermissionMode

    def configure(self):
        self.category = self.var('category')
        self.mode = self.var('mode')




@gws.ext.object.helper('storage')
class Object(gws.Node):
    provider: sqlite.Object
    permissions: t.List[Permission]

    def configure(self):
        self.permissions = self.root.create_many(Permission, self.var('permissions'))
        self.provider = self.root.create_required(sqlite.Object, self.config)

    def handle_action(self, req: gws.IWebRequester, p: Params, category: str) -> Response:
        readable, writable = self._category_permissions(category, req.user)

        data = None
        entries = []

        if p.verb == Verb.list:
            pass

        if p.verb == Verb.read:
            if not readable or not p.entryName:
                raise gws.base.web.error.Forbidden()
            rec = self.provider.read(category, p.entryName)
            if rec:
                data = gws.lib.json2.from_string(rec.data)

        if p.verb == Verb.write:
            if not writable or not p.entryName or not p.entryData:
                raise gws.base.web.error.Forbidden()
            d = gws.lib.json2.to_string(p.entryData)
            self.provider.write(category, p.entryName, d, req.user.uid)

        if p.verb == Verb.delete:
            if not writable or not p.entryName:
                raise gws.base.web.error.Forbidden()
            self.provider.delete(category, p.entryName)

        if readable:
            entries = [Entry(name=r.name) for r in self.provider.list(category)]

        return Response(
            data=data,
            directory=Directory(
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
                if p.mode == PermissionMode.all:
                    r = w = True
                elif p.mode == PermissionMode.read:
                    r = True
                elif p.mode == PermissionMode.write:
                    w = True

        return r, w