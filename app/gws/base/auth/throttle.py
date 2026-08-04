"""Authentication throttle.

Counts failed authentication attempts and blocks further attempts once a limit is reached.
Attempts are counted per remote address and, optionally, per login name.
"""

from typing import Optional

import gws
import gws.lib.datetimex as dtx
import gws.lib.sqlitex


class Config(gws.Config):
    """Authentication throttle options. (added in 8.4)"""

    maxAttemptsPerIp: int = 10
    """Failed attempts from one address before blocking."""
    maxAttemptsPerUser: int = 0
    """Failed attempts for one login from all addresses before blocking, 0=no limit."""
    windowTime: gws.Duration = '10m'
    """Time span in which failed attempts are counted."""
    blockTime: gws.Duration = '15m'
    """How long to block once the limit is reached."""
    allowFrom: Optional[list[str]]
    """Addresses exempt from throttling."""
    path: Optional[str]
    """Throttle storage path."""


_CLEANUP_INTERVAL = 600
_MAX_NAME_LENGTH = 128


class Object(gws.Node):
    """Authentication throttle."""

    maxAttemptsPerIp: int
    maxAttemptsPerUser: int
    windowTime: int
    blockTime: int
    allowFrom: set[str]
    dbPath: str

    table = 'throttle'

    def configure(self):
        self.maxAttemptsPerIp = self.cfg('maxAttemptsPerIp', default=10)
        self.maxAttemptsPerUser = self.cfg('maxAttemptsPerUser', default=0)
        self.windowTime = self.cfg('windowTime', default=dtx.parse_duration(Config.windowTime))
        self.blockTime = self.cfg('blockTime', default=dtx.parse_duration(Config.blockTime))
        self.allowFrom = set(self.cfg('allowFrom') or [])
        self.dbPath = self.cfg('path', default=f'{gws.c.MISC_DIR}/auth_throttle.sqlite')

        if self.blockTime <= self.windowTime:
            raise gws.ConfigurationError(f'invalid blockTime={self.blockTime}, must be greater than windowTime={self.windowTime}')

    ##

    def blocked_for(self, req: gws.WebRequester, method: gws.AuthMethod, credentials: gws.Data) -> int:
        """Return the time in seconds the given attempt remains blocked, 0 if it is allowed."""

        u_addr, u_user = self._get_uids(req, credentials)
        if not u_addr and not u_user:
            return 0

        now = gws.u.stime()
        rs = self._db().select(
            f'SELECT MAX(blocked_until) AS t FROM {self.table} WHERE uid IN (:u_addr, :u_user)',
            u_addr=u_addr,
            u_user=u_user,
        )

        t = rs[0]['t'] if rs else None
        return max(0, (t or 0) - now)

    def register(self, ok: bool, req: gws.WebRequester, method: gws.AuthMethod, credentials: gws.Data):
        """Register the outcome of an authentication attempt."""

        u_addr, u_user = self._get_uids(req, credentials)
        if not u_addr and not u_user:
            return

        if ok:
            self._db().execute(
                f'DELETE FROM {self.table} WHERE uid IN (:u_addr, :u_user)',
                u_addr=u_addr,
                u_user=u_user,
            )
            return

        if gws.u.stime() > self._cleanupTime + _CLEANUP_INTERVAL:
            self.cleanup()

        if u_addr:
            self._add_failure(u_addr, self.maxAttemptsPerIp)
        if u_user:
            self._add_failure(u_user, self.maxAttemptsPerUser)

    _cleanupTime = 0

    def cleanup(self):
        # a row may only be dropped when its window has elapsed *and* it holds no live block,
        # the row is the only place a block is recorded

        now = gws.u.stime()
        self._db().execute(
            f'DELETE FROM {self.table} WHERE first_time < :window_start AND blocked_until <= :now',
            window_start=now - self.windowTime,
            now=now,
        )
        self._cleanupTime = now

    ##

    def _add_failure(self, uid: str, max_attempts: int):
        now = gws.u.stime()

        # a new attempt starts a new window if the current one has elapsed.
        # an expired block needs no test of its own: since blockTime is greater than windowTime,
        # the window has always elapsed by the time a block runs out

        expired = 'first_time < :window_start'

        self._db().execute(
            f"""
                INSERT INTO {self.table} (uid, attempts, first_time, blocked_until)
                VALUES (:uid, 1, :now, 0)
                ON CONFLICT (uid) DO UPDATE SET
                    attempts      = CASE WHEN {expired} THEN 1    ELSE attempts + 1 END,
                    first_time    = CASE WHEN {expired} THEN :now ELSE first_time   END,
                    blocked_until = 0
            """,
            uid=uid,
            now=now,
            window_start=now - self.windowTime,
        )

        self._db().execute(
            f"""
                UPDATE {self.table} SET blocked_until = :until
                WHERE uid = :uid AND attempts >= :max_attempts
            """,
            uid=uid,
            until=now + self.blockTime,
            max_attempts=max_attempts,
        )

    def _get_uids(self, req: gws.WebRequester, credentials: gws.Data) -> tuple[str, str]:
        ip = req.ip
        if ip and ip in self.allowFrom:
            return '', ''

        u_addr = ''
        if ip and self.maxAttemptsPerIp > 0:
            u_addr = f'ip:{ip}'

        u_user = ''
        if self.maxAttemptsPerUser > 0:
            name = self._login_name(credentials)
            if name:
                u_user = f'user:{gws.u.sha256(name)}'

        return u_addr, u_user

    def _login_name(self, credentials: gws.Data) -> str:
        s = credentials.get('username')
        if not isinstance(s, str):
            return ''
        return s.strip().casefold()[:_MAX_NAME_LENGTH]

    ##

    _sqlitex: gws.lib.sqlitex.Object

    def _db(self):
        if getattr(self, '_sqlitex', None) is None:
            ddl = f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    uid           TEXT NOT NULL PRIMARY KEY,
                    attempts      INTEGER NOT NULL,
                    first_time    INTEGER NOT NULL,
                    blocked_until INTEGER NOT NULL
                )
            """
            self._sqlitex = gws.lib.sqlitex.Object(self.dbPath, ddl)
        return self._sqlitex
