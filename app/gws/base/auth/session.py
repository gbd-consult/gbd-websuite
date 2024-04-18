from typing import Optional

import datetime

import gws


class Object(gws.AuthSession):
    def __init__(
            self,
            uid: str,
            user: gws.User,
            method: Optional[gws.AuthMethod],
            data: dict = None,
            created: datetime.datetime = None,
            updated: datetime.datetime = None,
            is_changed = True,
    ):
        self.uid = uid
        self.method = method
        self.user = user
        self.data = data or {}
        self.created = created or datetime.datetime.now()
        self.updated = updated or datetime.datetime.now()
        self.isChanged = is_changed

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, val):
        self.data[key] = val
        self.isChanged = True
