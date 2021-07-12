import gws
import gws.types as t
import gws.lib.json2
from . import sqlite


class Error(gws.Error):
    """Generic storage error"""
    pass


class NotFound(Error):
    """Storage entry not found"""
    pass


class AccessDenied(Error):
    """No permission to read or write the entry"""
    pass


class Backend(t.Enum):
    sqlite = 'sqlite'


class Mode(t.Enum):
    read = 'read'  #: an object can be read
    write = 'write'  #: an object can be written and deleted
    all = 'all'  #: an object can be read and written


class PermissionRule(gws.WithAccess):
    """Permission rule for a storage category"""

    category: str  #: storage category name
    mode: Mode  #: allowed mode (read/write)


class Config(gws.WithType):
    """Storage helper"""

    backend: Backend  #: backend type
    path: t.Optional[str]  #: path to the storage file
    permissions: t.Optional[t.List[PermissionRule]]  #: permission rules



class StorageEntry(gws.Data):
    category: str
    name: str



class StorageDirectory(gws.Data):
    category: str
    writable: bool
    readable: bool
    entries: t.List[StorageEntry]



class StorageElement(gws.Data):
    entry: StorageEntry
    data: dict



class StorageRecord(gws.Data):
    category: str
    name: str
    user_fid: str
    data: str
    created: int
    updated: int


class Object(gws.Node):
    def __init__(self):
        super().__init__()
        self.backend = None
        self.permissions: t.List[PermissionRule] = []

    def configure(self):
        
        self.permissions = self.var('permissions', default=[])
        if self.var('backend') == 'sqlite':
            self.backend = self.root.create_unbound_object(sqlite.Object, self.config)

    def read(self, entry: StorageEntry, user: gws.IUser) -> StorageElement:
        if not self.can_read_category(entry.category, user):
            raise AccessDenied()
        rec: StorageRecord = self.backend.read(entry)
        if not rec:
            raise NotFound()
        return StorageElement(
            entry=StorageEntry(category=rec.category, name=rec.name),
            data=gws.lib.json2.from_string(rec.data))

    def write(self, entry: StorageEntry, user: gws.IUser, data: dict) -> StorageEntry:
        if not self.can_write_category(entry.category, user):
            raise AccessDenied()
        rec = self.backend.write(entry, user, gws.lib.json2.to_string(data))
        return StorageEntry(category=rec.category, name=rec.name)

    def dir(self, category: str, user: gws.IUser) -> t.List[gws.StorageEntry]:
        if not self.can_read_category(category, user):
            raise AccessDenied()
        return self.backend.dir(category)

    def delete(self, entry: StorageEntry, user: gws.IUser):
        if not self.can_write_category(entry.category, user):
            raise AccessDenied()
        self.backend.delete(entry.category, entry.name)

    def reset(self):
        self.backend.reset()

    def can_read_category(self, category: str, user: gws.IUser) -> bool:
        return self._can_use(category, user, Mode.read)

    def can_write_category(self, category: str, user: gws.IUser) -> bool:
        return self._can_use(category, user, Mode.write)

    def _can_use(self, category: str, user: gws.IUser, mode):
        for p in self.permissions:
            if p.category in (category, '*') and p.mode in (mode, Mode.all) and user.can_use(p):
                return True
        return False
