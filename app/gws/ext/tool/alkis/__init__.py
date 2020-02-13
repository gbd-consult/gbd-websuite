import gws
import gws.common.db
import gws.ext.db.provider.postgres
import gws.gis.feature
import gws.gis.shape

import gws.types as t

from .data import index, adresse, flurstueck
from .util import export
from .util.connection import AlkisConnection


class Config(t.WithType):
    """ALKIS tool."""

    db: str = ''  #: database provider ID
    crs: t.Crs  #: CRS for the alkis data
    dataSchema: str = 'public'  #: schema where ALKIS tables are stored, must be readable
    indexSchema: str = 'gws'  #: schema to store gws internal indexes, must be writable
    excludeGemarkung: t.Optional[t.List[str]]  #: Gemarkung (AU) IDs to exclude from search


class Gemarkung(t.Data):
    """Gemarkung (Administrative Unit) object"""

    gemarkung: str  #: Gemarkung name
    gemarkungUid: str  #: Gemarkung uid
    gemeinde: str  #: Gemeinde name
    gemeindeUid: str  #: Gemeinde uid


class BaseQuery(t.Data):
    gemarkung: str = ''
    gemarkungOrGemeindeUid: str = ''
    gemarkungUid: str = ''
    gemeinde: str = ''
    gemeindeUid: str = ''


class FindFlurstueckQuery(BaseQuery):
    withEigentuemer: bool = False
    withBuchung: bool = False

    bblatt: str = ''
    flaecheBis: str = ''
    flaecheVon: str = ''
    flurnummer: str = ''
    flurstuecksfolge: str = ''
    fsUids: t.List[str] = []
    hausnummer: str = ''
    name: str = ''
    nenner: str = ''
    strasse: str = ''
    vnum: str = ''
    vorname: str = ''
    zaehler: str = ''

    shape: t.IShape = None
    limit: str = 0


class FindFlurstueckResult(t.Data):
    features: t.List[t.IFeature] = []
    total: int = 0


class FindAdresseQuery(BaseQuery):
    bisHausnummer: str = ''
    hausnummer: str = ''
    hausnummerNotNull: t.Optional[bool]
    kreis: str = ''
    kreisUid: str = ''
    land: str = ''
    landUid: str = ''
    regierungsbezirk: str = ''
    regierungsbezirkUid: str = ''
    strasse: str = ''

    limit: int = 0


class FindAdresseResult(t.Data):
    features: t.List[t.IFeature] = []
    total: int = 0


class FindStrasseQuery(BaseQuery):
    pass


class FindStrasseResult(t.Data):
    strassen: t.List[str]


_COMBINED_FS_PARAMS = ['landUid', 'gemarkungUid', 'flurnummer', 'zaehler', 'nenner', 'flurstuecksfolge']
_COMBINED_AD_PARAMS = ['strasse', 'hausnummer', 'plz', 'gemeinde', 'bisHausnummer']

_COMBINED_PARAMS_DELIM = '_'


class Object(gws.Object):
    def __init__(self):
        super().__init__()

        self.crs = ''
        self.has_index = False
        self.has_source = False
        self.has_flurnummer = False
        self.connect_args = {}
        self.db = None

    def configure(self):
        super().configure()

        self.crs = self.var('crs')
        self.db: gws.ext.db.provider.postgres.Object = t.cast(
            gws.ext.db.provider.postgres.Object,
            gws.common.db.require_provider(self, 'gws.ext.db.provider.postgres'))

        self.connect_args = {
            'params': self.db.connect_params,
            'index_schema': self.var('indexSchema'),
            'data_schema': self.var('dataSchema'),
            'crs': self.crs,
            'exclude_gemarkung': self.var('excludeGemarkung')
        }

        with self.connect() as conn:
            if 'ax_flurstueck' in conn.table_names(self.connect_args['data_schema']):
                gws.log.info(f'ALKIS sources in "{self.db.uid}" found')
                self.has_source = True
            else:
                gws.log.info(f'ALKIS sources in "{self.db.uid}" NOT found')

            if index.ok(conn):
                gws.log.info(f'ALKIS indexes in "{self.db.uid}" found')
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

    def gemarkung_list(self) -> t.List[Gemarkung]:
        with self.connect() as conn:
            return [Gemarkung(r) for r in flurstueck.gemarkung_list(conn)]

    def find_flurstueck(self, query: FindFlurstueckQuery, **kwargs) -> FindFlurstueckResult:
        features = []

        q = self._query_to_dict(query)
        q.update(kwargs)
        q = self._remove_restricted_params(q)

        with self.connect() as conn:
            total, rs = flurstueck.find(conn, q)
            for rec in rs:
                rec = self._remove_restricted_data(q, rec)
                features.append(gws.gis.feature.Feature(
                    uid=rec['gml_id'],
                    attributes=rec,
                    shape=gws.gis.shape.from_wkb_hex(rec['geom'], self.crs)
                ))

        return FindFlurstueckResult(features=features, total=total)

    def find_flurstueck_combined(self, combined_param: str, **kwargs) -> FindFlurstueckResult:
        q = self._expand_combined_params(combined_param, _COMBINED_FS_PARAMS)
        q.update(kwargs)
        return self.find_flurstueck(FindFlurstueckQuery(q))

    def find_adresse(self, query: FindAdresseQuery, **kwargs) -> FindAdresseResult:
        features = []

        q = self._query_to_dict(query)
        q.update(kwargs)

        with self.connect() as conn:
            total, rs = adresse.find(conn, q)
            for rec in rs:
                features.append(gws.gis.feature.Feature(
                    uid=rec['gml_id'],
                    attributes=rec,
                    shape=gws.gis.shape.from_xy(rec['x'], rec['y'], self.crs)
                ))

        return FindAdresseResult(features=features, total=total)

    def find_adresse_combined(self, combined_param: str, **kwargs) -> FindAdresseResult:
        q = self._expand_combined_params(combined_param, _COMBINED_AD_PARAMS)
        q.update(kwargs)
        return self.find_adresse(FindAdresseQuery(q))

    def find_strasse(self, query: FindStrasseQuery, **kwargs) -> FindStrasseResult:
        q = self._query_to_dict(query)
        q.update(kwargs)
        with self.connect() as conn:
            ls = flurstueck.strasse_list(conn, q)
        return FindStrasseResult(strassen=ls)

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
        params = gws.extend(self.connect_args['params'], {
            'user': user,
            'password': password,
        })
        connect_args = gws.extend(self.connect_args, {'params': params})
        return AlkisConnection(**connect_args)

    def _query_to_dict(self, query):
        q = {k: v for k, v in gws.as_dict(query).items() if not gws.is_empty(v)}

        uid = q.pop('gemarkungOrGemeindeUid', None)
        if uid:
            uid = str(uid).split(':')
            if len(uid) == 2 and uid[0] == 'gemeinde':
                q['gemeindeUid'] = uid[1]
            if len(uid) == 2 and uid[0] == 'gemarkung':
                q['gemarkungUid'] = uid[1]

        return q
