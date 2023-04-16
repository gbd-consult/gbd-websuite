import gws
import gws.plugin.postgres.provider
import gws.gis.crs
import gws.base.feature
import gws.base.shape
import gws.types as t

from . import types
from .data import adresse, flurstueck, index
from .util.connection import AlkisConnection


class Config(gws.Config):
    """Basic ALKIS configuration"""

    dbUid: str = ''
    """database provider ID"""
    crs: gws.CrsName 
    """CRS for the ALKIS data"""
    dataSchema: str = 'public' 
    """schema where ALKIS tables are stored"""
    indexSchema: str = 'gws' 
    """schema to store GWS internal indexes"""
    excludeGemarkung: t.Optional[list[str]] 
    """Gemarkung (Administrative Unit) IDs to exclude from search"""


_COMBINED_FS_PARAMS = ['landUid', 'gemarkungUid', 'flurnummer', 'zaehler', 'nenner', 'flurstuecksfolge']
_COMBINED_AD_PARAMS = ['strasse', 'hausnummer', 'plz', 'gemeinde', 'bisHausnummer']

_COMBINED_PARAMS_DELIM = '_'


class Object(gws.Node):
    has_index = False
    has_source = False
    has_flurnummer = False
    crs: gws.ICrs
    connect_params: dict = {}
    data_schema = ''
    index_schema = ''

    def configure(self):

        self.crs = gws.gis.crs.get(self.cfg('crs'))
        self.index_schema = self.cfg('indexSchema')
        self.data_schema = self.cfg('dataSchema')

        db = gws.plugin.postgres.provider.require_for(self)

        self.connect_params = gws.merge(
            {},
            db.config,
            index_schema=self.index_schema,
            data_schema=self.data_schema,
            crs=self.crs,
            exclude_gemarkung=self.cfg('excludeGemarkung'),
        )

        with self.connection() as conn:
            if 'ax_flurstueck' in conn.table_names(self.data_schema):
                gws.log.debug(f'ALKIS sources in {db.uid!r} found')
                self.has_source = True
            else:
                gws.log.warning(f'ALKIS sources in {db.uid!r} NOT found')

            if index.ok(conn):
                gws.log.debug(f'ALKIS indexes in {db.uid!r} found')
                self.has_index = True
                self.has_flurnummer = flurstueck.has_flurnummer(conn)
            else:
                gws.log.warning(f'ALKIS indexes in {db.uid!r} NOT found')

    # public index tools

    def create_index(self, user, password):
        if not self.has_source:
            raise ValueError(f'ALKIS sources NOT found')
        with self.write_connection(user, password) as conn:
            index.create(conn, read_user=self.connect_params.get('user'))

    def drop_index(self, user, password):
        with self.write_connection(user, password) as conn:
            index.drop(conn)

    def index_ok(self):
        with self.connection() as conn:
            return index.ok(conn)

    # public search tools

    def gemarkung_list(self) -> list[types.Gemarkung]:
        with self.connection() as conn:
            return [types.Gemarkung(r) for r in flurstueck.gemarkung_list(conn)]

    def find_flurstueck(self, query: types.FindFlurstueckQuery, **kwargs) -> types.FindFlurstueckResult:

        features = []
        qdict = self._remove_restricted_params(gws.merge({}, query, kwargs))

        with self.connection() as conn:
            total, rs = flurstueck.find(conn, qdict)
            for rec in rs:
                rec = self._remove_restricted_data(qdict, rec)
                features.append(gws.base.feature.Feature(
                    uid=rec['gml_id'],
                    attributes=rec,
                    shape=gws.base.shape.from_wkb_hex(rec['geom'], self.crs)
                ))

        return types.FindFlurstueckResult(features=features, total=total)

    def find_flurstueck_combined(self, combined_param: str, **kwargs) -> types.FindFlurstueckResult:
        q = self._expand_combined_params(combined_param, _COMBINED_FS_PARAMS)
        return self.find_flurstueck(types.FindFlurstueckQuery(q), **kwargs)

    def find_adresse(self, query: types.FindAdresseQuery, **kwargs) -> types.FindAdresseResult:
        features = []

        with self.connection() as conn:
            total, rs = adresse.find(conn, gws.merge({}, query, kwargs))
            for rec in rs:
                features.append(gws.base.feature.Feature(
                    uid=rec['gml_id'],
                    attributes=rec,
                    shape=gws.base.shape.from_xy(rec['x'], rec['y'], self.crs)
                ))

        return types.FindAdresseResult(features=features, total=total)

    def find_adresse_combined(self, combined_param: str, **kwargs) -> types.FindAdresseResult:
        q = self._expand_combined_params(combined_param, _COMBINED_AD_PARAMS)
        return self.find_adresse(types.FindAdresseQuery(q), **kwargs)

    def find_strasse(self, query: types.FindStrasseQuery, **kwargs) -> types.FindStrasseResult:
        with self.connection() as conn:
            rs = flurstueck.strasse_list(conn, gws.merge({}, query, kwargs))
        return types.FindStrasseResult(strassen=[types.Strasse(r) for r in rs])

    ##

    def connection(self):
        return AlkisConnection(self.connect_params)

    def write_connection(self, user, password):
        params = gws.merge(self.connect_params, {
            'user': user,
            'password': password,
        })
        return AlkisConnection(params)

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


##

def create(root: gws.IRoot, cfg: gws.Config, parent: gws.Node = None, shared: bool = False) -> Object:
    key = gws.pick(cfg, 'db', 'crs', 'dataSchema', 'indexSchema', 'excludeGemarkung')
    return root.create(Object, cfg, parent, shared, key)
