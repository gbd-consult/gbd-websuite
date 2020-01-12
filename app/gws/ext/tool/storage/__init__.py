import gws
import gws.tools.json2

import gws.types as t

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


class Config(t.WithType):
    """Storage"""

    backend: Backend
    path: t.Optional[str]


#:export
class StorageEntry(t.Data):
    category: str
    name: str


#:export
class StorageElement(t.Data):
    entry: StorageEntry
    data: dict


#:export
class StorageRecord(t.Data):
    category: str
    name: str
    user_fid: str
    data: str
    created: int
    updated: int


class Object(gws.Object):
    def __init__(self):
        super().__init__()
        self.backend = None

    def configure(self):
        super().configure()
        if self.var('backend') == 'sqlite':
            self.backend = self.create_unbound_object(sqlite.Object, self.config)

    def read(self, entry: StorageEntry, user: t.IUser) -> StorageElement:
        if not self.can_read_category(entry.category, user):
            raise AccessDenied()
        rec: StorageRecord = self.backend.read(entry)
        if not rec:
            raise NotFound()
        return StorageElement(
            entry=StorageEntry(category=rec.category, name=rec.name),
            data=gws.tools.json2.from_string(rec.data))

    def write(self, entry: StorageEntry, user: t.IUser, data: dict) -> StorageEntry:
        if not self.can_write_category(entry.category, user):
            raise AccessDenied()
        rec = self.backend.write(entry, user, gws.tools.json2.to_string(data))
        return StorageEntry(category=rec.category, name=rec.name)

    def dir(self, category: str, user: t.IUser) -> t.List[t.StorageEntry]:
        if not self.can_read_category(category, user):
            return []
        return self.backend.dir(category)

    def reset(self):
        self.backend.reset()

    def can_read_category(self, category: str, user: t.IUser) -> bool:
        # @TODO granular permissions
        return True

    def can_write_category(self, category: str, user: t.IUser) -> bool:
        # @TODO granular permissions
        return True
