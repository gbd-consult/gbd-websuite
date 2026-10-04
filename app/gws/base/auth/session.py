"""Authentication session object."""

from typing import Optional

import datetime
import gws.lib.datetimex

import gws


class Object(gws.AuthSession):
    """Authentication session."""

    def __init__(
            self,
            uid: str,
            user: gws.User,
            method: Optional[gws.AuthMethod],
            data: dict = None,
            created: datetime.datetime = None,
            updated: datetime.datetime = None,
            is_changed=True,
            is_transient=False,
    ):
        """Create a session.

        Args:
            uid: Session uid.
            user: The session user.
            method: The method that created the session, ``None`` for the guest session.
            data: Session data.
            created: Creation time, defaults to now.
            updated: Last update time, defaults to now.
            is_changed: Whether the session has unsaved changes.
            is_transient: Whether the session lives only for the duration of the request.
        """
        self.uid = uid
        self.method = method
        self.user = user
        self.data = data or {}
        self.created = created or gws.lib.datetimex.now()
        self.updated = updated or gws.lib.datetimex.now()
        self.isChanged = is_changed
        self.isTransient = is_transient

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, val):
        self.data[key] = val
        self.isChanged = True
