import gws
import gws.types as t

from . import manager


class Config(gws.Config):
    """Database provider"""
    pass


class Object(gws.Node, gws.IDatabaseProvider):
    mgr: manager.Object

    def configure(self):
        self.mgr = self.cfg('_defaultManager')

    def session(self):
        return self.mgr.session(self)

    def table(self, table_name, columns=None, **kwargs):
        return self.mgr.table(self, table_name, columns, **kwargs)

    def describe(self, table_name):
        with self.session() as sess:
            return sess.describe(table_name)


def get_for(obj: gws.INode, uid: str = None, ext_type: str = None):
    mgr = obj.root.app.databaseMgr

    uid = uid or obj.cfg('dbUid')
    if uid:
        p = mgr.provider(uid)
        if not p:
            raise gws.Error(f'database provider {uid!r} not found')
        return p

    if obj.cfg('_defaultProvider'):
        return obj.cfg('_defaultProvider')

    ext_type = ext_type or obj.extType
    p = mgr.first_provider(ext_type)
    if not p:
        raise gws.Error(f'no database providers of type {ext_type!r} found')
    return p
