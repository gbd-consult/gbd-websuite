import re

import gws
import gws.config
import gws.gis.feature
import gws.gis.proj
import gws.gis.shape
import gws.gis.proj
import gws.tools.json2
import gws.tools.misc as misc
import gws.types as t
from .impl import Connection, Error


class Config(t.WithType):
    """Postgres/Postgis database provider"""

    database: str = ''  #: database name
    host: str = 'localhost'  #: database host
    password: str  #: password
    port: int = 5432  #: database port
    timeout: t.duration = 0  #: query timeout
    uid: str  #: unique id
    user: str  #: username


PING_TIMEOUT = 5


class Object(gws.Object, t.DbProviderObject):
    conn = None
    error = Error

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

    def connect(self, extra_connect_params=None):
        return Connection(gws.extend(self.connect_params, extra_connect_params))

    def configure(self):
        super().configure()

        def ping():
            p = self.connect_params
            p['connect_timeout'] = PING_TIMEOUT
            try:
                with Connection(p):
                    gws.log.info(f'db connection "{self.uid}": ok')
            except Error as e:
                raise gws.Error(f'cannot open db connection "{self.uid}"', e.args[0]) from e

        gws.get_global(f'db_ping_{self.uid}', ping)

    def geometry_props(self, table: t.SqlTableConfig):
        col = gws.get(table, 'geometryColumn')
        if not col:
            return [None, None]

        def props():
            with self.connect() as conn:
                return [
                    conn.crs_for_column(table.name, col),
                    conn.geometry_type_for_column(table.name, col)
                ]

        return gws.get_global('geometry_props.' + table.name + '.' + col, props)

    def select(self, args: t.SelectArgs, extra_connect_params=None):

        with self.connect(extra_connect_params) as conn:

            where = []
            parms = []

            search_col = args.table.get('searchColumn')
            geom_col = args.table.get('geometryColumn')
            key_col = args.table.get('keyColumn')

            kw = args.get('keyword')
            if kw and search_col:
                kw = kw.lower().replace('%', '').replace('_', '')
                where.append(f'{conn.quote_ident(search_col)} ILIKE %s')
                parms.append('%' + kw + '%')

            crs = None
            if geom_col:
                # @TODO cache me
                crs = conn.crs_for_column(args.table.name, geom_col)

            shape = args.get('shape')
            if shape and geom_col:
                if shape.type == 'Point':
                    shape = shape.tolerance_buffer(args.get('tolerance'))

                shape = shape.transform(crs)

                where.append(f'ST_Intersects(ST_SetSRID(%s::geometry,%s), "{geom_col}")')
                parms.append(shape.wkb_hex)
                parms.append(crs.split(':')[1])

            ids = args.get('ids')
            if ids:
                ph = ','.join(['%s'] * len(ids))
                where.append(f'{conn.quote_ident(key_col)} IN ({ph})')
                parms.extend(ids)

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

            features = []

            for rec in recs:
                shape = None
                if geom_col:
                    g = rec.pop(geom_col, None)
                    if g:
                        shape = gws.gis.shape.from_wkb(g, crs)

                pk = (rec.get(key_col) if key_col else None) or gws.random_string(16)

                features.append(gws.gis.feature.Feature({
                    'uid': misc.sha256(args.table.name) + '_' + str(pk),
                    'attributes': rec,
                    'shape': shape
                }))

            return features

    def update(self, table: t.SqlTableConfig, recs: t.List[dict]):
        ids = []
        self._prepare_for_update(table, recs)

        with self.connect() as conn:
            with conn.transaction():
                for rec in recs:
                    id = rec.pop(table.keyColumn)
                    conn.update(table.name, table.keyColumn, id, rec)
                    ids.append(id)

        return ids

    def insert(self, table: t.SqlTableConfig, recs: t.List[dict]):
        ids = []
        self._prepare_for_update(table, recs)

        with self.connect() as conn:
            with conn.transaction():
                for rec in recs:
                    ids.append(conn.insert_one(table.name, table.keyColumn, rec))

        return ids

    def delete(self, table: t.SqlTableConfig, recs: t.List[dict]):
        ids = [rec.pop(table.keyColumn) for rec in recs]

        with self.connect() as conn:
            with conn.transaction():
                conn.delete_many(table.name, table.keyColumn, ids)

        return ids

    def _prepare_for_update(self, table, recs):
        crs, geometry_type = self.geometry_props(table)
        srid = gws.gis.proj.as_srid(crs) if crs else None

        # @TODO: support EWKB directly

        geom_col = gws.get(table, 'geometryColumn')
        if geom_col:
            for rec in recs:
                if geom_col in rec:
                    geom_val = rec[geom_col]
                    if isinstance(geom_val, gws.gis.shape.Shape) and crs:
                        geom_val.transform(crs)
                        ph = 'ST_SetSRID(%s::geometry,%s)'
                        if geometry_type.startswith('MULTI'):
                            ph = f'ST_Multi({ph})'
                        rec[geom_col] = [ph, geom_val.wkb_hex, srid]
