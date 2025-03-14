import gws

import gws.lib.sa as sa


class Object(gws.DatabaseConnection):
    def __init__(self, conn):
        self.saConnection = conn

    def begin(self):
        return self.saConnection.begin()

    def commit(self):
        return self.saConnection.commit()

    def rollback(self):
        return self.saConnection.rollback()

    def exec(self, statement, **params):
        if isinstance(statement, str):
            statement = sa.text(statement)
        return self.saConnection.execute(statement, params)

    def exec_commit(self, statement, **params):
        if isinstance(statement, str):
            statement = sa.text(statement)
        res = self.saConnection.execute(statement, params)
        self.saConnection.commit()
        return res

    def fetch_all(self, statement, **params):
        return [dict(r) for r in self.exec(statement, **params)]

    def fetch_one(self, statement, **params):
        rs = self.fetch_all(statement, **params)
        return rs[0] if rs else None
