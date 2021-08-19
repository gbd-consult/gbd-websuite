import gws
import gws.types as t


class Session(gws.IAuthSession):
    def __init__(self, typ: str, user: gws.IUser, method: t.Optional[gws.IAuthMethod], uid=None, data=None):
        self.changed = False
        self.data = data or {}
        self.method = method
        self.typ = typ
        self.uid = uid
        self.user = user

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, val):
        self.data[key] = val
        self.changed = True
