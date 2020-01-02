import re

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


def create_shared(obj: t.IObject, cfg) -> 'Object':
    key = '-'.join([
        f'h={cfg.host}',
        f'p={cfg.port}',
        f'u={cfg.user}',
        f'd={cfg.database}'
    ])
    prov: Object = obj.create_shared_object(
        'gws.ext.db.provider.postgres',
        gws.as_uid(key),
        cfg)
    return prov


PING_TIMEOUT = 5


class Object(gws.common.db.provider.Sql):
    error = driver.Error

    @property
    def connect_params(self):
        params = {
            'application_name': 'gws',
        }
        for p in 'host', 'port', 'user', 'password', 'database':
            params[p] = self.var(p)
        timeout = self.var('timeout')
        if timeout:
            # statement_timeout is in ms
            params['options'] = '-c statement_timeout=%d' % (timeout * 1000)
        return params

    def connect(self, extra_connect_params=None) -> driver.Connection:
        return driver.Connection(gws.extend(self.connect_params, extra_connect_params))

    def configure(self):
        super().configure()

        def ping():
            p = self.connect_params
            p['connect_timeout'] = PING_TIMEOUT
            try:
                with driver.Connection(p):
                    gws.log.info(f'db connection "{self.uid}": ok')
            except driver.Error as e:
                raise gws.Error(f'cannot open db connection "{self.uid}"', e.args[0]) from e

        gws.get_global(f'db_ping_{self.uid}', ping)

    def describe(self, table: t.SqlTable):
        def f():
            with self.connect() as conn:
                return {c['name']: t.SqlTableColumn(c) for c in conn.columns(table.name)}

        key = 'gws.ext.provider.postgres.describe.' + table.name
        return gws.get_global(key, f)

    def select(self, args: t.SelectArgs, extra_connect_params=None) -> t.List[t.IFeature]:

        with self.connect(extra_connect_params) as conn:

            where = []
            parms = []

            search_col = args.table.search_column
            geom_col = args.table.geometry_column
            key_col = args.table.key_column
            crs = args.table.geometry_crs

            kw = args.get('keyword')
            if kw and search_col:
                kw = kw.lower().replace('%', '').replace('_', '')
                where.append(f'{conn.quote_ident(search_col)} ILIKE %s')
                parms.append('%' + kw + '%')

            shape = args.get('shape')
            if shape and geom_col:
                if shape.type == 'Point':
                    shape = shape.tolerance_buffer(args.get('tolerance'))

                shape = shape.transformed(crs)

                where.append(f'ST_Intersects(ST_SetSRID(%s::geometry,%s), "{geom_col}")')
                parms.append(shape.wkb_hex)
                parms.append(crs.split(':')[1])

            uids = args.get('uids')
            if uids:
                if not key_col:
                    return []
                ph = ','.join(['%s'] * len(uids))
                where.append(f'{conn.quote_ident(key_col)} IN ({ph})')
                parms.extend(uids)

            if args.get('extraWhere'):
                where.append('(%s)' % args.extraWhere.replace('%', '%%'))

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

            sql = f'SELECT * FROM {conn.quote_table(args.table.name)} WHERE {where} {sort} {limit}'

            gws.log.debug(f'SELECT_FEATURES_START {sql} p={parms}')
            recs = list(r for r in conn.select(sql, parms))
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
        table = t.SqlTable({
            'name': cfg.get('name'),
            'geometry_column': '',
            'geometry_crs': '',
            'geometry_type': '',
            'key_column': '',
            'search_column': ''
        })

        cols = self.describe(table)

        s = cfg.get('keyColumn')
        if not s:
            cs = [c.name for c in cols.values() if c.is_key]
            if len(cs) != 1:
                raise gws.Error(f'invalid primary key for table {table.name!r}')
            s = cs[0]
        table.key_column = s

        s = cfg.get('geometryColumn')
        if not s:
            cs = [c.name for c in cols.values() if c.is_geometry]
            if cs:
                gws.log.info(f'found geometry column {cs[0]!r} for table {table.name!r}')
                s = cs[0]
        table.geometry_column = s

        if table.geometry_column:
            col = cols[table.geometry_column]
            table.geometry_crs = col.crs
            table.geometry_type = col.type

        table.search_column = cfg.get('searchColumn')

        return table

    def auto_data_model(self, table: t.SqlTable) -> t.IModel:
        rules = []
        for name, col in self.describe(table).items():
            if col.is_geometry or col.is_key:
                continue
            rules.append(t.ModelRule({
                'title': name,
                'name': name,
                'source': name,
                'type': col.type,
            }))
        m: t.IModel = self.create_object('gws.common.model', t.Config({'rules': rules}))
        return m

    def _feature_to_record(self, table: t.SqlTable, feature: t.IFeature) -> dict:
        rec = {a.name: a.value for a in feature.attributes}

        if table.key_column:
            rec[table.key_column] = feature.uid

        if not table.geometry_column or not feature.shape:
            return rec

        # @TODO: support EWKB directly

        shape = feature.shape.transformed(table.geometry_crs)
        ph = 'ST_SetSRID(%s::geometry,%s)'
        if table.geometry_type.startswith('multi'):
            ph = f'ST_Multi({ph})'
        rec[table.geometry_column] = [ph, shape.wkb_hex, table.geometry_crs.split(':')[1]]

        return rec

    def _record_to_feature(self, table: t.SqlTable, rec: dict) -> t.IFeature:
        shape = None
        if table.geometry_column:
            g = rec.pop(table.geometry_column, None)
            if g:
                shape = gws.gis.shape.from_wkb(g, table.geometry_crs)

        uid = None
        if table.key_column:
            uid = str(rec.pop(table.key_column, None))
        if not uid:
            uid = gws.random_string(16)

        return gws.gis.feature.Feature(uid=uid, attributes=rec, shape=shape)

    def _get_by_uids(self, table, uids):
        return self.select(t.SelectArgs({
            'table': table,
            'uids': list(uids),
        }))
