import gws
import gws.config
import gws.gis.feature
import gws.gis.proj
import gws.gis.shape
import gws.tools.json2
import gws.tools.misc as misc
import gws.types as t
from .impl import Connection, Error


class Config(t.WithType):
    """Postgis database provider"""

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

            if args.get('sort'):
                sort = 'ORDER BY %s' % args.get('sort')
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
                    shape = gws.gis.shape.from_wkb(
                        rec.pop(geom_col),
                        crs)

                pk = (rec.get(key_col) if key_col else None) or gws.random_string(16)

                features.append(gws.gis.feature.Feature({
                    'uid': misc.sha256(args.table.name) + '_' + str(pk),
                    'attributes': rec,
                    'shape': shape
                }))

            return features
