import re

import gws
import gws.types as t
import gws.base.db
import gws.base.model
import gws.config
import gws.lib.feature
import gws.lib.proj
import gws.lib.proj
import gws.lib.shape
import gws.lib.json2

from . import driver

_DESCRIBE_CACHE_LIFETIME = 3600

_ext_class = 'gws.ext.db.provider.postgres'


def shared_provider(root: gws.RootObject, cfg) -> 'Object':
    key = '-'.join([
        f'h={cfg.host}',
        f'p={cfg.port}',
        f'u={cfg.user}',
        f'd={cfg.database}'
    ])
    return t.cast('Object', root.create_shared_object(_ext_class, gws.as_uid(key), cfg))


def require_provider(obj: gws.INode) -> 'Object':
    uid = obj.var('db')
    if uid:
        prov = obj.root.find(klass=_ext_class, uid=uid)
        if not prov:
            raise gws.Error(f'{obj.uid}: db provider {uid!r} not found')
    else:
        prov = obj.root.find(klass=_ext_class)
        if not prov:
            raise gws.Error(f'{obj.uid}: db provider {_ext_class!r} not found')
    return t.cast('Object', prov)


@gws.ext.Config('db.provider.postgres')
class Config(gws.Config):
    """Postgres/Postgis database provider"""

    database: str = ''  #: database name
    host: str = 'localhost'  #: database host
    password: str  #: password
    port: int = 5432  #: database port
    timeout: gws.Duration = '0'  #: query timeout
    connectTimeout: gws.Duration = '0'  #: connect timeout
    user: str  #: username


@gws.ext.Object('db.provider.postgres')
class Object(gws.Node, gws.ISqlDbProvider):
    error = driver.Error

    @property
    def connect_params(self):
        params = {
            'application_name': 'gws',
        }
        for p in 'host', 'port', 'user', 'password', 'database':
            params[p] = self.var(p)
        p = self.var('connectTimeout')
        if p:
            params['connect_timeout'] = p
        p = self.var('timeout')
        if p:
            # statement_timeout is in ms
            params['options'] = '-c statement_timeout={p * 1000}'
        return params

    def connect(self, extra_connect_params=None) -> driver.Connection:
        return driver.Connection(gws.merge(self.connect_params, extra_connect_params))

    def configure(self):

        def ping():
            gws.log.debug(f'db: ping {self.uid!r}')
            try:
                with driver.Connection(self.connect_params):
                    gws.log.debug(f'db connection "{self.uid}": ok')
            except driver.Error as e:
                raise gws.Error(f'cannot open db connection "{self.uid}"', e.args[0]) from e

        gws.get_global(f'db_ping_{self.uid}', ping)

    def describe(self, table: gws.SqlTable) -> t.Dict[str, gws.SqlTableColumn]:
        def f():
            gws.log.debug(f'db: describe {key!r}')
            with self.connect() as conn:
                return {c['name']: gws.SqlTableColumn(c) for c in conn.columns(table.name)}

        key = _ext_class + '.describe.' + table.name
        return gws.get_cached_object(key, f, _DESCRIBE_CACHE_LIFETIME)

    def select(self, args: gws.SqlSelectArgs, extra_connect_params=None) -> t.List[gws.IFeature]:

        with self.connect(extra_connect_params) as conn:

            where = []
            values = []

            search_col = args.table.search_column
            geom_col = args.table.geometry_column
            key_col = args.table.key_column
            crs = args.table.geometry_crs

            kw = args.keyword
            if kw and search_col:
                # @TODO search mode (startsWith, contains, exact etc)
                kw = kw.lower().replace('%', '').replace('_', '')
                where.append(f'{conn.quote_ident(search_col)} ILIKE %s')
                values.append('%' + kw + '%')

            shape = args.shape
            if shape and geom_col:
                shape = shape.tolerance_polygon(args.map_tolerance).transformed_to(crs)
                where.append(f'ST_Intersects(%s::geometry, "{geom_col}")')
                values.append(shape.ewkb_hex)

            uids = args.uids
            if uids:
                if not key_col:
                    return []
                ph = ','.join(['%s'] * len(uids))
                where.append(f'{conn.quote_ident(key_col)} IN ({ph})')
                values.extend(uids)

            if args.extra_where:
                where.append('(' + args.extra_where[0] + ')')
                values.extend(args.extra_where[1:])

            if not where:
                return []

            where_str = ' AND '.join(where)

            sort = limit = ''

            p = args.get('sort')
            if p:
                if re.match(r'^\w+$', p):
                    p = conn.quote_ident(p)
                sort = 'ORDER BY %s' % p

            if args.limit:
                limit = 'LIMIT %d' % args.limit

            cols = '*'
            if args.columns:
                cols = ','.join(args.columns)

            sql = f'SELECT {cols} FROM {conn.quote_table(args.table.name)} WHERE {where_str} {sort} {limit}'

            gws.log.debug(f'SELECT_FEATURES_START {sql} p={values}')
            recs = list(r for r in conn.select(sql, values))
            gws.log.debug(f'SELECT_FEATURES_END len={len(recs)}')

            return [self._record_to_feature(args.table, r) for r in recs]

    def edit_operation(self, operation: str, table: gws.SqlTable, features: t.List[gws.IFeature]) -> t.List[gws.IFeature]:
        uids = []
        recs = [self._feature_to_record(table, f) for f in features]

        if operation == 'insert':
            with self.connect() as conn:
                with conn.transaction():
                    for rec in recs:
                        uids.append(conn.insert_one(table.name, table.key_column, rec, with_id=True))

            return self._get_by_uids(table, uids)

        if operation == 'update':

            with self.connect() as conn:
                with conn.transaction():
                    for rec in recs:
                        conn.update(table.name, table.key_column, rec)
                        uids.append(rec.get(table.key_column))

            return self._get_by_uids(table, uids)

        if operation == 'delete':
            uids = [f.uid for f in features]
            with self.connect() as conn:
                with conn.transaction():
                    conn.delete_many(table.name, table.key_column, uids)

            return []

    def configure_table(self, cfg: gws.base.db.SqlTableConfig) -> gws.SqlTable:
        table = gws.SqlTable(
            name=cfg.get('name'),
            geometry_column=None,
            geometry_crs=None,
            geometry_type=None,
            key_column=None,
            search_column=None
        )

        cols = self.describe(table)

        if not cols:
            raise gws.Error(f'table {table.name!r} not found or not accessible')

        cname = cfg.get('keyColumn')
        if cname:
            if cname not in cols:
                raise gws.Error(f'invalid keyColumn {cname!r} for table {table.name!r}')
        else:
            cs = [c.name for c in cols.values() if c.is_key]
            if len(cs) == 1:
                cname = cs[0]
                gws.log.debug(f'found primary key {cname!r} for table {table.name!r}')
            else:
                gws.log.warn(f'invalid primary key for table {table.name!r} found={cs}')

        table.key_column = cname

        cname = cfg.get('geometryColumn')
        if cname:
            if cname not in cols or not cols[cname].geom_type or not cols[cname].crs:
                raise gws.Error(f'invalid geometryColumn {cname!r} for table {table.name!r}')
        else:
            cs = [c.name for c in cols.values() if c.is_geometry]
            if cs:
                gws.log.debug(f'found geometry column {cs[0]!r} for table {table.name!r}')
                cname = cs[0]

        if cname:
            table.geometry_column = cname
            table.geometry_crs = cols[cname].crs
            table.geometry_type = cols[cname].geom_type

        cname = cfg.get('searchColumn')

        if cname:
            if cname not in cols:
                raise gws.Error(f'invalid searchColumn {cname!r} for table {table.name!r}')
            table.search_column = cname

        return table

    def table_data_model_config(self, table: gws.SqlTable) -> gws.Config:
        rules = []

        for name, col in self.describe(table).items():
            if col.is_geometry:
                continue
            rules.append(gws.base.model.Rule(
                title=name,
                name=name,
                source=name,
                type=col.type,
                editable=not col.is_key,
            ))

        return gws.Config(
            rules=rules,
            geometryType=table.geometry_type,
            crs=table.geometry_crs,
        )

    def _feature_to_record(self, table: gws.SqlTable, feature: gws.IFeature) -> dict:
        rec = {a.name: a.value for a in feature.attributes}

        if table.key_column:
            rec[table.key_column] = feature.uid

        if table.geometry_type and feature.shape:
            shape = feature.shape.to_type(table.geometry_type).transformed_to(table.geometry_crs)
            rec[table.geometry_column] = shape.ewkb_hex

        return rec

    def _record_to_feature(self, table: gws.SqlTable, rec: dict) -> gws.IFeature:
        shape = None
        if table.geometry_column:
            g = rec.pop(table.geometry_column, None)
            if g:
                # assuming geometries are returned in hex
                shape = gws.lib.shape.from_wkb_hex(g, table.geometry_crs)

        uid = None
        if table.key_column:
            uid = str(rec.get(table.key_column, None))
        if not uid:
            uid = gws.random_string(16)

        return gws.lib.feature.new(uid=uid, attributes=rec, shape=shape)

    def _get_by_uids(self, table, uids):
        return self.select(gws.SqlSelectArgs({
            'table': table,
            'uids': list(uids),
        }))
