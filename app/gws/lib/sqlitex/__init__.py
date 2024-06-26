"""Convenience wrapper for the SA SQLite engine.

This wrapper accepts a database path and optionally an "init" DDL statement.
It executes queries given in a text form.

Error recovery:

- if a query fails and there's no table in the database, the "init" DDL is executed, and the query is repeated.
- If the database happens to be locked, it waits for a while and then attempts to repeat the query.

"""

from typing import Optional
import gws
import gws.lib.sa as sa


class Error(gws.Error):
    pass


class Object:
    saEngine: sa.Engine

    def __init__(self, db_path: str, init_ddl: Optional[str] = ''):
        self.dbPath = db_path
        self.initDdl = init_ddl

    def execute(self, stmt: str, **params):
        """Execute a text DML statement and commit."""

        self._exec2(False, stmt, params)

    def select(self, stmt: str, **params) -> list[dict]:
        """Execute a text select statement and commit."""

        return self._exec2(True, stmt, params)

    def insert(self, table_name: str, rec: dict):
        """Insert a new record (dict) into a table."""

        keys = ','.join(rec)
        vals = ','.join(':' + k for k in rec)

        self._exec2(False, f'INSERT INTO {table_name} ({keys}) VALUES({vals})', rec)

    ##

    def _exec2(self, is_select, stmt, params):
        while True:
            sa_exc = None
            try:
                with self._engine().connect() as conn:
                    if is_select:
                        return [gws.u.to_dict(r) for r in conn.execute(sa.text(stmt), params)]
                    conn.execute(sa.text(stmt), params)
                    conn.commit()
                    return
            except sa.Error as exc:
                sa_exc = exc

            # @TODO using strings for error checking, is there a better way?

            s = str(sa_exc)

            if 'no such table' in s and self.initDdl:
                gws.log.warning(f'sqlitex: error={s}, running init...')
                try:
                    with self._engine().connect() as conn:
                        conn.execute(sa.text(self.initDdl))
                        conn.commit()
                    continue
                except sa.Error as exc:
                    sa_exc = exc

            if 'database is locked' in s:
                gws.log.warning(f'sqlitex: error={s}, waiting...')
                gws.u.sleep(0.1)
                continue

            raise gws.Error('sqlitex error') from sa_exc

    def _engine(self):
        if getattr(self, 'saEngine', None) is None:
            self.saEngine = sa.create_engine(f'sqlite:///{self.dbPath}', poolclass=sa.NullPool, echo=True)
        return self.saEngine
