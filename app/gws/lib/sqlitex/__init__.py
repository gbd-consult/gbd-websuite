"""Convenience wrapper for the SA SQLite engine.

This wrapper accepts a database path and optionally an "init" DDL statement.
It executes queries given in a text form.

Failed queries are repeated up to 3 times to work around transient errors, like the DB being locked.

If the error message is "no such table", the wrapper runs the  "init" DDL before repeating.

"""

from typing import Optional

import gws
import gws.lib.sa as sa


class Error(gws.Error):
    pass


class Object:
    saEngine: sa.Engine

    def __init__(self, db_path: str, init_ddl: Optional[str] = '', connect_args: Optional[dict] = None):
        self.dbPath = db_path
        self.initDDL = init_ddl
        self.connectArgs = connect_args or {}

    def execute(self, stmt: str, **params):
        """Execute a text DML statement and commit."""

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
        self._exec2(False, f'UPDATE {table_name} SET {vals} WHERE uid=:uid', {'uid': uid, **rec})

    def delete(self, table_name: str, uid):
        """Delete a record by uid from a table."""

        self._exec2(False, f'DELETE FROM {table_name} WHERE uid=:uid', {'uid': uid})

        ##

    _MAX_ERRORS = 3
    _SLEEP_TIME = 0.1

    def _exec2(self, is_select, stmt, params):
        err_cnt = 0

        while True:
            sa_exc = None

            try:
                with self._engine().connect() as conn:
                    if is_select:
                        return [gws.u.to_dict(r) for r in conn.execute(sa.text(stmt), params)]
                    conn.execute(sa.text(stmt), params)
                    conn.commit()
                    return []
            except sa.Error as exc:
                sa_exc = exc

            # @TODO using strings for error checking, is there a better way?

            if 'no such table' in str(sa_exc) and self.initDDL:
                gws.log.warning(f'sqlitex: {self.dbPath}: {sa_exc}, running init...')
                try:
                    with self._engine().connect() as conn:
                        conn.execute(sa.text(self.initDDL))
                        conn.commit()
                    continue
                except sa.Error as exc:
                    sa_exc = exc

            if 'database is locked' in str(sa_exc):
                gws.log.warning(f'sqlitex: {self.dbPath}: locked, waiting...')
                gws.u.sleep(self._SLEEP_TIME)
                continue

            err_cnt += 1
            if err_cnt < self._MAX_ERRORS:
                gws.log.warning(f'sqlitex: {self.dbPath}: {sa_exc}, waiting...')
                gws.u.sleep(self._SLEEP_TIME)
                continue

            raise gws.Error(f'sqlitex: {self.dbPath}: {sa_exc}') from sa_exc

    def _engine(self):
        if getattr(self, 'saEngine', None) is None:
            self.saEngine = sa.create_engine(
                f'sqlite:///{self.dbPath}',
                poolclass=sa.NullPool,
                echo=False,
                connect_args=self.connectArgs,
            )
        return self.saEngine
