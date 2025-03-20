import gws

import gws.lib.sa as sa


class Object(gws.DatabaseConnection, sa.Connection):
    def exec(self, sql, **params):
        if isinstance(sql, str):
            sql = sa.text(sql)
        return self.execute(sql, params)

    def exec_commit(self, sql, **params):
        if isinstance(sql, str):
            sql = sa.text(sql)
        res = self.execute(sql, params)
        self.commit()
        return res

    def get_all(self, sql, **params):
        return [r._asdict() for r in self.exec(sql, **params)]

    def get_first(self, sql, **params):
        res = self.exec(sql, **params)
        r = res.first()
        return r._asdict() if r else None

    def get_scalars(self, sql, **params):
        res = self.exec(sql, **params)
        return res.scalars().all()

    def get_scalar(self, sql, **params):
        res = self.exec(sql, **params)
        return res.scalar()

