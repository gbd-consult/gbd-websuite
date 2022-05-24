import gws
import gws.types as t


def require_provider(obj: t.IObject, klass='gws.ext.db.provider') -> t.IDbProvider:
    uid = obj.var('db')
    if uid:
        prov = obj.root.find('gws.ext.db.provider', uid)
        if not prov:
            raise gws.Error(f'{obj.uid}: db provider {uid!r} not found')
    else:
        prov = obj.root.find_first(klass)
        if not prov:
            raise gws.Error(f'{obj.uid}: db provider {klass!r} not found')
    return t.cast(t.IDbProvider, prov)


class SqlTableConfig(t.Config):
    """SQL database table"""

    name: str  #: table name
    geometryColumn: t.Optional[str]  #: geometry column name
    keyColumn: t.Optional[str]  #: primary key column name
    searchColumn: t.Optional[str]  #: column to be searched for


#:export
class SqlTable(t.Data):
    name: str
    key_column: str = ''
    search_column: str = ''
    geometry_column: str = ''
    geometry_type: t.GeometryType = ''
    geometry_crs: t.Crs = ''


#:export
class SelectArgs(t.Data):
    extra_where: t.Optional[list]
    keyword: t.Optional[str]
    keyword_columns: t.Optional[t.List[str]]
    limit: t.Optional[int]
    map_tolerance: t.Optional[float]
    shape: t.Optional[t.IShape]
    sort: t.Optional[str]
    table: t.SqlTable
    uids: t.Optional[t.List[str]]
    columns: t.Optional[t.List[str]]
    depth: t.Optional[int]


#:export
class SqlTableColumn(t.Data):
    name: str
    type: t.AttributeType
    geom_type: t.GeometryType
    native_type: str
    crs: t.Crs
    is_key: bool
    is_geometry: bool
