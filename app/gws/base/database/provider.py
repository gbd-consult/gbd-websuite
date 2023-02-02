import gws
import gws.types as t

from . import sql, manager


class Config(gws.Config):
    """Database provider"""

    database: str = ''
    """database name"""
    host: str = 'localhost'
    """database host"""
    password: str
    """password"""
    port: int = 5432
    """database port"""
    timeout: gws.Duration = '0'
    """query timeout"""
    connectTimeout: gws.Duration = '0'
    """connect timeout"""
    username: str
    """username"""


class Object(gws.Node, gws.IDatabaseProvider):
    mgr: manager.Object

    def configure(self):
        self.mgr = self.var('_manager')

    def session(self):
        return self.mgr.session(self)

    def table(self, name, columns=None, **kwargs):
        return self.mgr.table(self, name, columns, **kwargs)


def get_for(obj: gws.INode, uid: str = None, ext_type: str = None) -> gws.IDatabaseProvider:
    uid = uid or obj.var('db')
    if not uid and obj.var('_provider'):
        return obj.var('_provider')

    mgr = obj.root.app.databaseMgr
    ext_type = ext_type or obj.extType

    if uid:
        p = mgr.provider(uid, ext_type)
        if not p:
            raise gws.Error(f'database provider {ext_type!r} {uid!r} not found')
        return p

    p = mgr.first_provider(ext_type)
    if not p:
        raise gws.Error(f'database provider {ext_type!r} not found')
    return p
