import gws.types as t


#:export ISession
class Session(t.ISession):
    def __init__(self, type, user: t.IUser, method: t.IAuthMethod = None, uid=None, data=None):
        self.changed = False
        self.data: dict = data or {}
        self.method: t.IAuthMethod = method
        self.type: str = type
        self.uid: str = uid
        self.user: t.IUser = user

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, val):
        self.data[key] = val
        self.changed = True
