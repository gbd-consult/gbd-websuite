import gws
import gws.base.db.postgres
import gws.lib.feature
import gws.lib.shape
import gws.lib.crs
import gws.types as t

from . import types
from .data import adresse, flurstueck, index
from .util.connection import AlkisConnection


class Config(gws.Config):
    """Basic ALKIS configuration"""

    db: str = ''  #: database provider ID
    crs: gws.CrsId  #: CRS for the ALKIS data
    dataSchema: str = 'public'  #: schema where ALKIS tables are stored
    indexSchema: str = 'gws'  #: schema to store GWS internal indexes
    excludeGemarkung: t.Optional[t.List[str]]  #: Gemarkung (Administrative Unit) IDs to exclude from search


_COMBINED_FS_PARAMS = ['landUid', 'gemarkungUid', 'flurnummer', 'zaehler', 'nenner', 'flurstuecksfolge']
_COMBINED_AD_PARAMS = ['strasse', 'hausnummer', 'plz', 'gemeinde', 'bisHausnummer']

_COMBINED_PARAMS_DELIM = '_'


class Object(gws.Node):
    db: gws.base.db.postgres.provider.Object
    has_index = False
    has_source = False
    has_flurnummer = False
    crs: gws.ICrs
    connect_args: t.Dict = {}
    data_schema = ''
    index_schema = ''

    def configure(self):

        self.crs = gws.lib.crs.get(self.var('crs'))
        self.db = gws.base.db.postgres.provider.require_for(self)

        self.index_schema = self.var('indexSchema')
        self.data_schema = self.var('dataSchema')

        self.connect_args = {
            'params': self.db.connect_params,
            'index_schema': self.index_schema,
            'data_schema': self.data_schema,
            'crs': self.crs,
            'exclude_gemarkung': self.var('excludeGemarkung')
        }

        with self.connect() as conn:
            if 'ax_flurstueck' in conn.table_names(self.connect_args['data_schema']):
                gws.log.debug(f'ALKIS sources in "{self.db.uid}" found')
                self.has_source = True
            else:
                gws.log.warn(f'ALKIS sources in "{self.db.uid}" NOT found')

            if index.ok(conn):
                gws.log.debug(f'ALKIS indexes in "{self.db.uid}" found')
                self.has_index = True
                self.has_flurnummer = flurstueck.has_flurnummer(conn)
            else:
                gws.log.warn(f'ALKIS indexes in "{self.db.uid}" NOT found')

    # public index tools

    def create_index(self, user, password):
        if not self.has_source:
            raise ValueError(f'ALKIS sources in "{self.db.uid}" NOT found')
        with self._connect_for_writing(user, password) as conn:
            index.create(conn, read_user=self.connect_args['params']['user'])

    def drop_index(self, user, password):
        with self._connect_for_writing(user, password) as conn:
            index.drop(conn)

    def index_ok(self):
        with self.connect() as conn:
            return index.ok(conn)

    # public search tools

    def gemarkung_list(self) -> t.List[types.Gemarkung]:
        with self.connect() as conn:
            return [types.Gemarkung(r) for r in flurstueck.gemarkung_list(conn)]

    def find_flurstueck(self, query: types.FindFlurstueckQuery, **kwargs) -> types.FindFlurstueckResult:
        features = []

        q = self._query_to_dict(query)
        q.update(kwargs)
        q = self._remove_restricted_params(q)

        with self.connect() as conn:
            total, rs = flurstueck.find(conn, q)
            for rec in rs:
                rec = self._remove_restricted_data(q, rec)
                features.append(gws.lib.feature.Feature(
                    uid=rec['gml_id'],
                    attributes=rec,
                    shape=gws.lib.shape.from_wkb_hex(rec['geom'], self.crs)
                ))

        return types.FindFlurstueckResult(features=features, total=total)

    def find_flurstueck_combined(self, combined_param: str, **kwargs) -> types.FindFlurstueckResult:
        q = self._expand_combined_params(combined_param, _COMBINED_FS_PARAMS)
        q.update(kwargs)
        return self.find_flurstueck(types.FindFlurstueckQuery(q))

    def find_adresse(self, query: types.FindAdresseQuery, **kwargs) -> types.FindAdresseResult:
        features = []

        q = self._query_to_dict(query)
        q.update(kwargs)

        with self.connect() as conn:
            total, rs = adresse.find(conn, q)
            for rec in rs:
                features.append(gws.lib.feature.Feature(
                    uid=rec['gml_id'],
                    attributes=rec,
                    shape=gws.lib.shape.from_xy(rec['x'], rec['y'], self.crs)
                ))

        return types.FindAdresseResult(features=features, total=total)

    def find_adresse_combined(self, combined_param: str, **kwargs) -> types.FindAdresseResult:
        q = self._expand_combined_params(combined_param, _COMBINED_AD_PARAMS)
        q.update(kwargs)
        return self.find_adresse(types.FindAdresseQuery(q))

    def find_strasse(self, query: types.FindStrasseQuery, **kwargs) -> types.FindStrasseResult:
        q = self._query_to_dict(query)
        q.update(kwargs)
        with self.connect() as conn:
            rs = flurstueck.strasse_list(conn, q)
        return types.FindStrasseResult(strassen=[types.Strasse(r) for r in rs])

    def connect(self):
        return AlkisConnection(**self.connect_args)

    ##

    def _expand_combined_params(self, value, fields) -> dict:
        q = {}
        for val, field in zip(value.split(_COMBINED_PARAMS_DELIM), fields):
            if val and val != '0':
                q[field] = val
        return q

    def _remove_restricted_params(self, q):
        if not q.get('withEigentuemer'):
            q.pop('vorname', None)
            q.pop('name', None)
        if not q.get('withBuchung'):
            q.pop('bblatt', None)
        return q

    def _remove_restricted_data(self, q, rec):
        if q.get('withEigentuemer'):
            return rec

        if q.get('withBuchung'):
            for b in rec.get('buchung', []):
                b.pop('eigentuemer', None)
            return rec

        rec.pop('buchung', None)
        return rec

    def _connect_for_writing(self, user, password):
        params = gws.merge(self.connect_args['params'], {
            'user': user,
            'password': password,
        })
        connect_args = gws.merge(self.connect_args, {'params': params})
        return AlkisConnection(**connect_args)

    def _query_to_dict(self, query):
        return {k: v for k, v in gws.to_dict(query).items() if not gws.is_empty(v)}


##

def create(root: gws.IRoot, cfg: gws.Config, parent: gws.Node = None, shared: bool = False) -> Object:
    key = gws.pick(cfg, 'db', 'crs', 'dataSchema', 'indexSchema', 'excludeGemarkung')
    return root.create_object(Object, cfg, parent, shared, key)
