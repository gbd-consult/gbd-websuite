import sqlalchemy.orm as saorm
import sqlalchemy.sql as sql

import gws
import gws.types as t


class Object(gws.IDatabaseSession):

    def __init__(self, provider: gws.IDatabaseProvider, sess: saorm.Session):
        self.provider = provider
        self.sa = sess

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        self.sa.close()

    def begin(self):
        return self.sa.begin()

    def commit(self):
        return self.sa.commit()

    def rollback(self):
        return self.sa.rollback()

    def execute(self, stmt: sql.Executable, params=None, **kwargs):
        return self.sa.execute(stmt, params, **kwargs)

    def describe(self, table_name: str):
        return self.provider.mgr.describe(self, table_name)

    def autoload(self, table_name: str):
        return self.provider.mgr.autoload(self, table_name)
