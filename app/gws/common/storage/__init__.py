import gws
import gws.types as t


#:export
class StorageEntry(t.Data):
    category: str
    name: str


#:export IStorage
class Object(gws.Object, t.IStorage):
    def read(self, entry: StorageEntry, user: t.IUser) -> dict:
        pass

    def write(self, entry: StorageEntry, user: t.IUser, data: dict) -> bool:
        pass

    def dir(self, user: t.IUser) -> t.List[t.StorageEntry]:
        pass

    def can_read(self, r, user: t.IUser) -> bool:
        pass

    def can_write(self, r, user: t.IUser) -> bool:
        pass
