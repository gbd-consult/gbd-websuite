import re
import time

import gws
import gws.config
import gws.gis.feature
import gws.gis.proj
import gws.gis.shape
import gws.gis.proj
import gws.common.db.provider
import gws.tools.json2

import gws.types as t

from . import driver

_DESCRIBE_CACHE_LIFETIME = 3600


def create_shared(root: t.IRootObject, cfg) -> 'Object':
    key = '-'.join([
        f'h={cfg.host}',
        f'p={cfg.port}',
        f'u={cfg.user}',
        f'd={cfg.database}'
    ])
    return t.cast(Object, root.create_shared_object(
        'gws.ext.db.provider.postgres',
        gws.as_uid(key),
        cfg))


class Object(gws.common.db.provider.Sql):
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
        super().configure()

        def ping():
            attempts = 10
            pause = 1

            for a in range(1, attempts + 1):
                try:
                    with driver.Connection(self.connect_params):
                        gws.log.debug(f'db: ping {self.uid!r} attempt {a} ok')
                    return 1
                except driver.Error as e:
                    gws.log.debug(f'db: ping {self.uid!r} attempt {a} FAILED {e!r}')
                time.sleep(pause)

            raise gws.Error(f'cannot open db connection "{self.uid}"')

        gws.get_global(f'db_ping_{self.uid}', ping)

    def describe(self, table: t.SqlTable) -> t.Dict[str, t.SqlTableColumn]:
        def f():
            gws.log.debug(f'db: describe {key!r}')
            with self.connect() as conn:
                return {c['name']: t.SqlTableColumn(c) for c in conn.columns(table.name)}

        key = 'gws.ext.provider.postgres.describe.' + table.name
        return gws.get_cached_object(key, f, _DESCRIBE_CACHE_LIFETIME)

    def select(self, args: t.SelectArgs, extra_connect_params=None) -> t.List[t.IFeature]:

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
                where.append(f'POSITION(%s IN LOWER({conn.quote_ident(search_col)})) > 0')
                values.append(kw.lower())

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

            where = ' AND '.join(where)

            sort = limit = ''

            p = args.get('sort')
            if p:
                if re.match(r'^\w+$', p):
                    p = conn.quote_ident(p)
                sort = 'ORDER BY %s' % p

            if args.get('limit'):
                limit = 'LIMIT %d' % args.get('limit')

            cols = '*'
            if args.get('columns'):
                cols = ','.join(args.columns)

            sql = f'SELECT {cols} FROM {conn.quote_table(args.table.name)} WHERE {where} {sort} {limit}'

            gws.log.debug(f'SELECT_FEATURES_START {sql} p={values}')
            recs = list(r for r in conn.select(sql, values))
            gws.log.debug(f'SELECT_FEATURES_END len={len(recs)}')

            return [self._record_to_feature(args.table, r) for r in recs]

    def edit_operation(self, operation: str, table: t.SqlTable, features: t.List[t.IFeature]) -> t.List[t.IFeature]:
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

    def configure_table(self, cfg: gws.common.db.SqlTableConfig) -> t.SqlTable:
        table = t.SqlTable(
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

    def table_data_model_config(self, table: t.SqlTable) -> t.Config:
        cfg = {
            'rules': [],
            'geometryType': table.geometry_type,
            'crs': table.geometry_crs,
        }

        for name, col in self.describe(table).items():
            if col.is_geometry:
                continue
            cfg['rules'].append(t.ModelRule(
                title=name,
                name=name,
                source=name,
                type=col.type,
                editable=not col.is_key,
            ))

        return t.Config(cfg)

    def _feature_to_record(self, table: t.SqlTable, feature: t.IFeature) -> dict:
        rec = {a.name: a.value for a in feature.attributes}

        if table.key_column:
            rec[table.key_column] = feature.uid

        if table.geometry_column and feature.shape:
            shape = feature.shape.to_type(table.geometry_type).transformed_to(table.geometry_crs)
            rec[table.geometry_column] = shape.ewkb_hex

        return rec

    def _record_to_feature(self, table: t.SqlTable, rec: dict) -> t.IFeature:
        shape = None
        if table.geometry_column:
            g = rec.pop(table.geometry_column, None)
            if g:
                # assuming geometries are returned in hex
                shape = gws.gis.shape.from_wkb_hex(g, table.geometry_crs)

        uid = None
        if table.key_column:
            uid = str(rec.get(table.key_column, None))
        if not uid:
            uid = gws.random_string(16)

        return gws.gis.feature.Feature(uid=uid, attributes=rec, shape=shape)

    def _get_by_uids(self, table, uids):
        return self.select(t.SelectArgs({
            'table': table,
            'uids': list(uids),
        }))
