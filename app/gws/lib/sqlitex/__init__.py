"""Convenience wrapper for the SQLite driver.

This wrapper accepts a database path and optionally an "init" DDL script.
It executes queries given in a text form.

Each query runs on its own connection, which is closed immediately afterwards.

If a query fails with "no such table", the wrapper runs the "init" script and repeats the query once.
The script can contain multiple statements.

A query that fails with a recoverable error is repeated on a new connection.

"""

import sqlite3

import gws

BUSY_TIMEOUT = 5.0
"""Time in seconds to wait for a lock before giving up."""

MAX_ATTEMPTS = 3
"""How many times to repeat a query that failed with a recoverable error."""

SLEEP_TIME = 0.1
"""Time in seconds to wait between the attempts."""

_RECOVERABLE_ERRORS = {
    'SQLITE_BUSY',
    'SQLITE_CANTOPEN',
    'SQLITE_LOCKED',
    'SQLITE_PROTOCOL',
}
"""Errors worth repeating, matched against the leading part of the sqlite error name."""


class Error(gws.Error):
    pass


class Object:
    def __init__(self, db_path: str, init_ddl: str = '', uid_column: str = 'uid'):
        self.dbPath = db_path
        self.initDDL = init_ddl
        self.uidName = uid_column

    def execute(self, stmt: str, **params):
        """Execute a text DML statement."""

        self._exec2(False, stmt, params)

    def select(self, stmt: str, **params) -> list[dict]:
        """Execute a text select statement."""

        return self._exec2(True, stmt, params)

    def insert(self, table_name: str, rec: dict):
        """Insert a new record (dict) into a table."""

        keys = ','.join(rec)
        vals = ','.join(':' + k for k in rec)

        self._exec2(False, f'INSERT INTO {table_name} ({keys}) VALUES({vals})', rec)

    def update(self, table_name: str, rec: dict, uid):
        """Update a record (dict) in a table."""

        vals = ','.join(f'{k}=:{k}' for k in rec)
        self._exec2(
            False,
            f'UPDATE {table_name} SET {vals} WHERE {self.uidName}=:__uid',
            {'__uid': uid, **rec},
        )

    def delete(self, table_name: str, uid):
        """Delete a record by uid from a table."""

        self._exec2(
            False,
            f'DELETE FROM {table_name} WHERE {self.uidName}=:__uid',
            {'__uid': uid},
        )

    ##

    def _exec2(self, is_select, stmt, params):
        attempt = 0

        while True:
            attempt += 1
            try:
                return self._exec3(is_select, stmt, params)
            except sqlite3.Error as exc:
                gws.log.warning(f'sqlitex: {self.dbPath}: {exc}, sql={" ".join(stmt.split())}')
                name = getattr(exc, 'sqlite_errorname', '')
                if not any(name.startswith(e) for e in _RECOVERABLE_ERRORS) or attempt >= MAX_ATTEMPTS:
                    raise Error(f'sqlitex: {self.dbPath}: {exc}') from exc
                gws.u.sleep(SLEEP_TIME)

    def _exec3(self, is_select, stmt, params):
        conn = None

        try:
            conn = sqlite3.connect(self.dbPath, timeout=BUSY_TIMEOUT, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                return self._exec4(conn, is_select, stmt, params)
            except sqlite3.OperationalError as exc:
                if not self.initDDL or 'no such table' not in str(exc):
                    raise
                gws.log.warning(f'sqlitex: {self.dbPath}: {exc}, running init...')
                conn.executescript(self.initDDL)
                return self._exec4(conn, is_select, stmt, params)
        finally:
            if conn:
                conn.close()

    def _exec4(self, conn, is_select, stmt, params):
        cur = conn.execute(stmt, params)
        if is_select:
            return [dict(r) for r in cur]
        return []
