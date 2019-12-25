import gws
import gws.types as t

from . import provider


def configure_table(target: t.Object, prov: provider.Object) -> t.SqlTable:
    table = t.SqlTable({'name': target.var('table.name')})
    cols = prov.describe(table)

    s = target.var('table.keyColumn')
    if not s:
        cs = [c.name for c in cols.values() if c.is_key]
        if len(cs) != 1:
            raise gws.Error(f'invalid primary key for table {table.name!r}')
        s = cs[0]
    table.key_column = s

    s = target.var('table.geometryColumn')
    if not s:
        cs = [c.name for c in cols.values() if c.is_geometry]
        if cs:
            gws.log.info(f'found geometry column {cs[0]!r} for table {table.name!r}')
            s = cs[0]
    table.geometry_column = s

    table.search_column = target.var('table.searchColumn')

    if table.geometry_column:
        col = cols[table.geometry_column]
        table.geometry_crs = col.crs
        table.geometry_type = col.type

    return table


def configure_auto_data_model(target: t.Object, prov: provider.Object, table: t.SqlTable) -> t.DataModelObject:
    rules = []
    for name, col in prov.describe(table).items():
        if col.is_geometry or col.is_key:
            continue
        rules.append(t.DataModelRule({
            'title': name,
            'name': name,
            'source': name,
            'type': col.type,
        }))
    cfg = t.DataModelConfig({'rules': rules})
    return t.cast(t.DataModelObject, target.create_object('gws.common.datamodel', cfg))
