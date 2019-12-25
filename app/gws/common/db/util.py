import gws
import gws.types as t


def require_provider(obj: t.Object, klass='gws.ext.db.provider') -> t.DbProviderObject:
    s = obj.var('db')
    prov: t.DbProviderObject
    if s:
        prov = obj.root.find('gws.ext.db.provider', s)
    else:
        prov = obj.root.find_first(klass)
    if not prov:
        raise gws.Error(f'{obj.uid}: db provider not found')
    return prov
