from typing import Optional
import gws
import threading

import gws.lib.sa as sa


class Object(gws.DatabaseConnection):
    db: gws.DatabaseProvider
    saConn: sa.Connection

    def __init__(self, db: gws.DatabaseProvider, conn: sa.Connection):
        self.db = db
        self.saConn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        getattr(self.db, '_close_connection')()

    def execute(self, stmt, params=None):
        return self.saConn.execute(stmt, params)

    def commit(self):
        self.saConn.commit()

    def rollback(self):
        self.saConn.rollback()

    def exec(self, sql, **params):
        if isinstance(sql, str):
            sql = sa.text(sql)
        return self.saConn.execute(sql, params)

    def exec_commit(self, sql, **params):
        if isinstance(sql, str):
            sql = sa.text(sql)
        res = self.saConn.execute(sql, params)
        self.saConn.commit()
        return res

    def exec_rollback(self, sql, **params):
        if isinstance(sql, str):
            sql = sa.text(sql)
        res = self.saConn.execute(sql, params)
        self.saConn.rollback()
        return res

    def fetch_all(self, stmt, **params):
        return [r._asdict() for r in self.exec_rollback(stmt, **params)]

    def fetch_first(self, stmt, **params):
        res = self.exec_rollback(stmt, **params)
        r = res.first()
        return r._asdict() if r else None

    def fetch_scalars(self, stmt, **params):
        res = self.exec_rollback(stmt, **params)
        return list(res.scalars().all())

    def fetch_strings(self, stmt, **params):
        res = self.exec_rollback(stmt, **params)
        return [_to_str(s) for s in res.scalars().all()]

    def fetch_ints(self, stmt, **params):
        res = self.exec_rollback(stmt, **params)
        return [_to_int(s) for s in res.scalars().all()]

    def fetch_scalar(self, stmt, **params):
        res = self.exec_rollback(stmt, **params)
        return res.scalar()

    def fetch_string(self, stmt, **params):
        res = self.exec_rollback(stmt, **params)
        s = res.scalar()
        return _to_str(s) if s is not None else None

    def fetch_int(self, stmt, **params):
        res = self.exec_rollback(stmt, **params)
        s = res.scalar()
        return _to_int(s) if s is not None else None


##


def _to_int(s) -> int:
    if isinstance(s, int):
        return s
    raise ValueError(f'db: expected int, got {s=}')


def _to_str(s) -> str:
    if s is None:
        return ''
    return str(s)
