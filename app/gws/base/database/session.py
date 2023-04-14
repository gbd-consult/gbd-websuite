import gws
import gws.lib.sa as sa


class Object(gws.IDatabaseSession):

    def __init__(self, provider: gws.IDatabaseProvider):
        self.provider = provider
        self.saSession = sa.orm.Session(
            provider.engine(),
            autoflush=False,
            autobegin=False
        )
        self.nested = 0

    def __enter__(self):
        gws.log.debug(f'session enter: {self.provider.uid!r}: {self.nested=}')
        if self.nested == 0:
            self.saSession.begin()
        self.nested += 1
        return self

    def __exit__(self, type_, value, traceback):
        self.nested -= 1
        if self.nested == 0:
            self.saSession.close()
        gws.log.debug(f'session exit: {self.provider.uid!r}: {self.nested=}')

    def begin(self):
        return self.saSession.begin()

    def commit(self):
        return self.saSession.commit()

    def rollback(self):
        return self.saSession.rollback()

    def execute(self, stmt: sa.Executable, params=None, **kwargs):
        return self.saSession.execute(stmt, params, **kwargs)

    def describe(self, table_name: str):
        return self.provider.mgr.describe(self, table_name)

    def autoload(self, table_name: str):
        return self.provider.mgr.autoload(self, table_name)
