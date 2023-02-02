import sqlalchemy.engine.reflection
import sqlalchemy.orm as orm
import sqlalchemy.sql as sql

import gws
import gws.types as t


class Object(gws.IDatabaseSession):
    saSession: orm.Session
    provider: gws.IDatabaseProvider

    def __init__(self, provider: gws.IDatabaseProvider, sess: orm.Session):
        self.provider = provider
        self.saSession = sess

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        self.saSession.close()

    def begin(self):
        return self.saSession.begin()

    def commit(self):
        return self.saSession.commit()

    def rollback(self):
        return self.saSession.rollback()

    def execute(self, stmt: sql.Executable, params=None, **kwargs):
        return self.saSession.execute(stmt, params, **kwargs)

    def describe(self, table_name: str):
        return self.provider.mgr.describe(self, table_name)

    def autoload(self, table_name: str):
        return self.provider.mgr.autoload(self, table_name)
