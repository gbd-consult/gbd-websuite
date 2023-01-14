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

    def describe_table(self, table_name: str):
        return self.mgr.describe_table(self, self.qualified_table_name(table_name))

    def session(self):
        return self.mgr.session(self)

    def sa_make_table(self, name: str, columns: t.List[sql.sa.Column], **kwargs):
        return self.mgr.sa_make_table(self, name, columns, **kwargs)


def configure_for(obj: gws.INode, ext_type: str = None):
    return obj.root.app.databaseMgr.provider_for(obj, ext_type)
