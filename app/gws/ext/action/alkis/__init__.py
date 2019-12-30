import os
import re

import gws
import gws.tools.date
import gws.tools.job
import gws.config
import gws.gis.feature
import gws.gis.shape
import gws.common.printer.service
import gws.common.printer.types
import gws.common.template
import gws.ext.db.provider.postgres
import gws.tools.misc
import gws.types as t
import gws.tools.json2
import gws.web
from .data import index, adresse, flurstueck
from .tools import connection, export

"""
log table:


create table <name> (
    id serial primary key,
    app_name varchar(255),
    date_time timestamp,
    ip varchar(255),
    login varchar(255),
    user_name varchar(255),
    control_input varchar(255),
    control_result integer,
    fs_count integer,
    fs_ids text
)

grant insert on <name> to <user>
grant usage on <name>_id_seq to <user>

"""


class EigentuemerConfig:
    """Access to the Eigentümer (owner) information"""

    access: t.Access  #: access rights

    controlMode: bool = False  #: restricted mode enabled
    controlRules: t.Optional[t.List[str]]  #: list of regular expression for the restricted input control

    logTable: str = ''  #: data access protocol table name


class BuchungConfig:
    """Access to the Grundbuch (register) information"""

    access: t.Access  #: access rights


class UiConfig:
    """Flurstückssuche UI configuration."""

    useExport: bool = False  #: export function enabled
    useSelect: bool = False  #: select mode enabled
    usePick: bool = False  #: pick mode enabled
    searchSelection: bool = False  #: search in selection enabled
    searchSpatial: bool = False  #: spatial search enabled
    gemarkungListMode: str = 'combined'  #: plain = only "gemarkung", combined = "gemarkung(gemeinde)", tree = "gemeinde", then "gemarkung"
    autoSpatialSearch: bool = False  #: activate spatial search after submit


class Config(t.WithTypeAndAccess):
    """Flurstückssuche (cadaster parlcels search) action"""

    db: t.Optional[str]  #: database provider ID
    alkisSchema: str = 'public'  #: schema where ALKIS tables are stored, must be readable
    indexSchema: str = 'gws'  #: schema to store gws internal indexes, must be writable

    eigentuemer: t.Optional[EigentuemerConfig]  #: access to the Eigentümer (owner) information
    buchung: t.Optional[BuchungConfig]  #: access to the Grundbuch (register) information

    excludeGemarkung: t.Optional[t.List[str]]  #: Gemarkung (AU) IDs to exclude from search

    limit: int = 100  #: search results limit

    featureFormat: t.Optional[gws.common.template.FeatureFormatConfig]  #: template for on-screen Flurstueck details
    printTemplate: t.Optional[t.ext.template.Config]  #: template for printed Flurstueck details

    disableApi: bool = False  #: disable external access to this extension

    ui: t.Optional[UiConfig]  #: ui options


class FsGemarkung(t.Data):
    """Gemarkung (Administrative Unit) object"""

    gemarkung: str  #: Gemarkung name
    gemarkungUid: str  #: Gemarkung uid
    gemeinde: str  #: Gemeinde name
    gemeindeUid: str  #: Gemeinde uid


class FsSetupParams(t.Params):
    pass


class FsSetupResponse(t.Response):
    withEigentuemer: bool
    withBuchung: bool
    withControl: bool
    withFlurnummer: bool
    gemarkungen: t.List[FsGemarkung]
    printTemplate: t.TemplateProps
    limit: int
    ui: UiConfig


class FsStrassenParams(t.Params):
    gemarkungUid: t.Optional[str]
    gemeindeUid: t.Optional[str]
    gemarkungOrGemeindeUid: t.Optional[str]


class FsStrassenResponse(t.Response):
    strassen: t.List[str]


_COMBINED_FS_PARAMS = ['landUid', 'gemarkungUid', 'flurnummer', 'zaehler', 'nenner', 'flurstuecksfolge']
_COMBINED_AD_PARAMS = ['strasse', 'hausnummer', 'plz', 'gemeinde', 'bisHausnummer']

_COMBINED_PARAMS_DELIM = '_'


class FsQueryParams(t.Params):
    wantEigentuemer: t.Optional[bool]
    controlInput: t.Optional[str]
    fsUids: t.Optional[t.List[str]]

    bblatt: t.Optional[str]
    flaecheBis: t.Optional[str]
    flaecheVon: t.Optional[str]
    gemarkungUid: t.Optional[str]
    gemeindeUid: t.Optional[str]
    gemarkungOrGemeindeUid: t.Optional[str]
    hausnummer: t.Optional[str]
    name: t.Optional[str]
    strasse: t.Optional[str]
    vorname: t.Optional[str]
    vnum: t.Optional[str]
    flurnummer: t.Optional[str]
    zaehler: t.Optional[str]
    nenner: t.Optional[str]
    flurstuecksfolge: t.Optional[str]
    shapes: t.Optional[t.List[t.ShapeProps]]

    # combined fs search param (_COMBINED_FS_PARAMS joined by '_')
    alkisFs: t.Optional[str]
    # combined address search param (_COMBINED_AD_PARAMS joined by '_')
    alkisAd: t.Optional[str]


class FsAddressQueryParams(t.Params):
    land: t.Optional[str]
    regierungsbezirk: t.Optional[str]
    kreis: t.Optional[str]
    gemeinde: t.Optional[str]
    gemarkung: t.Optional[str]
    landUid: t.Optional[str]
    regierungsbezirkUid: t.Optional[str]
    kreisUid: t.Optional[str]
    gemarkungUid: t.Optional[str]
    gemeindeUid: t.Optional[str]
    gemarkungOrGemeindeUid: t.Optional[str]
    strasse: t.Optional[str]
    hausnummer: t.Optional[str]
    bisHausnummer: t.Optional[str]
    hausnummerNotNull: t.Optional[bool]

    alkisAd: t.Optional[str]


class FsPrintParams(FsQueryParams):
    printParams: t.Optional[gws.common.printer.types.PrintParams]
    highlightStyle: t.StyleProps


class FsExportParams(FsQueryParams):
    groups: t.List[str]


class FsDetailsParams(FsQueryParams):
    pass


class FsSearchResponse(t.Response):
    features: t.List[t.FeatureProps]
    total: int


class FsDetailsResponse(t.Response):
    feature: t.FeatureProps


class FsExportResponse(t.Response):
    url: str


class GeocoderAddress(t.Data):
    gemeinde: str = ''
    gemarkung: str = ''
    strasse: str = ''
    hausnummer: str = ''


_GEOCODER_ADDR_KEYS = 'gemeinde', 'gemarkung', 'strasse', 'hausnummer'


class GeocoderParams(t.Data):
    adressen: t.List[GeocoderAddress]
    crs: t.Crs


class GeocoderResponse(t.Response):
    coordinates: t.List[t.Point]


_cwd = os.path.dirname(__file__)

DEFAULT_FORMAT = gws.common.template.FeatureFormatConfig({
    'title': gws.common.template.Config({
        'type': 'html',
        'text': '{attributes.vollnummer}'
    }),
    'teaser': gws.common.template.Config({
        'type': 'html',
        'text': 'Flurstück {attributes.vollnummer}'
    }),
    'description': gws.common.template.Config({
        'type': 'html',
        'path': _cwd + '/templates/data.cx.html'
    })
})

DEFAULT_PRINT_TEMPLATE = gws.common.template.Config({
    'type': 'html',
    'path': _cwd + '/templates/print.cx.html',
    'pageWidth': 210,
    'pageHeight': 297,
    'mapWidth': 100,
    'mapHeight': 100,
    'qualityLevels': [
        t.TemplateQualityLevel({
            'dpi': 150
        })
    ]
})


class Object(gws.Object):
    def __init__(self):
        super().__init__()
        self.crs = ''
        self.has_index = False
        self.limit = 0
        self.short_feature_format: t.IFormat= None
        self.long_feature_format: t.IFormat = None
        self.print_template: t.ITemplate = None
        self.eigentuemer: EigentuemerConfig = None
        self.control_mode = False
        self.log_table = ''
        self.buchung: BuchungConfig = None
        self.disableApi = False
        self.connect_args = {}
        self.control_rules = []


    def configure(self):
        super().configure()

        p = self.var('db')
        db: gws.ext.db.provider.postgres.Object = self.root.find(
            'gws.ext.db.provider', p) if p else self.root.find_first('gws.ext.db.provider.postgres')

        self.crs = self._get_alkis_crs(db)

        self.connect_args = {
            'params': db.connect_params,
            'index_schema': self.var('indexSchema'),
            'data_schema': self.var('alkisSchema'),
            'crs': self.crs,
            'exclude_gemarkung': self.var('excludeGemarkung')
        }

        if self.index_ok():
            gws.log.info(f'ALKIS indexes in "{db.uid}" are fine')
            self.has_index = True
        else:
            gws.log.warn(f'ALKIS indexes in "{db.uid}" are not ok, please reindex')

        if not self.has_index:
            return

        self.limit = int(self.var('limit'))

        fmt = self.var('featureFormat') or gws.common.template.FeatureFormatConfig()
        for f in 'title', 'teaser', 'description':
            if not fmt.get(f):
                setattr(fmt, f, DEFAULT_FORMAT.get(f))

        self.short_feature_format = self.add_child('gws.common.format', t.Config({
            'title': fmt.title,
            'teaser': fmt.teaser,
        }))

        self.long_feature_format = self.add_child('gws.common.format', t.Config({
            'title': fmt.title,
            'teaser': fmt.teaser,
            'description': fmt.description,
        }))

        self.print_template = self.add_child(
            'gws.ext.template',
            self.var('printTemplate', default=DEFAULT_PRINT_TEMPLATE))

        self.buchung = self.var('buchung')

        self.eigentuemer = self.var('eigentuemer')
        self.control_mode = False
        self.log_table = None

        if self.eigentuemer:
            self.log_table = self.eigentuemer.get('logTable')
            if self.eigentuemer.get('controlMode'):
                self.control_mode = True
                self.control_rules = self.eigentuemer.get('controlRules') or []

        if self.log_table:
            with self._connect() as conn:
                if not conn.user_can('INSERT', self.log_table):
                    raise ValueError(f'no INSERT acccess to {self.log_table!r}')

        self.disableApi = self.var('disableApi')
        if self.disableApi:
            gws.log.info(f'Alkis API disabled for {self.uid!r}')

    def api_fs_setup(self, req: t.IRequest, p: FsSetupParams) -> FsSetupResponse:
        """Return project-specific Flurstueck-Search settings"""

        self._precheck_request(req, p.projectUid)

        with self._connect() as conn:
            return FsSetupResponse({
                'withEigentuemer': self._can_read_eigentuemer(req),
                'withControl': self._can_read_eigentuemer(req) and self.control_mode,
                'withBuchung': self._can_read_buchung(req),
                'withFlurnummer': flurstueck.has_flurnummer(conn),
                'gemarkungen': flurstueck.gemarkung_list(conn),
                'printTemplate': self.print_template.props,
                'limit': self.limit,
                'ui': self.var('ui'),
            })

    def api_fs_strassen(self, req: t.IRequest, p: FsStrassenParams) -> FsStrassenResponse:
        """Return a list of Strassen for the given Gemarkung"""

        self._precheck_request(req, p.projectUid)
        self._set_gemeinde_or_gemarkung(p)

        with self._connect() as conn:
            return FsStrassenResponse({
                'strassen': flurstueck.strasse_list(conn, p),
            })

    def api_fs_search(self, req: t.IRequest, p: FsQueryParams) -> FsSearchResponse:
        """Perform a Flurstueck search"""

        self._precheck_request(req, p.projectUid)
        self._set_gemeinde_or_gemarkung(p)

        total, fs = self._fetch_and_format(req, p, self.short_feature_format)

        if total > self.limit:
            raise gws.web.error.Conflict()

        return FsSearchResponse({
            'total': total,
            'features': sorted(fs, key=lambda p: p.title)
        })

    def api_fs_details(self, req: t.IRequest, p: FsDetailsParams) -> FsDetailsResponse:
        """Return a Flurstueck feature with details"""

        self._precheck_request(req, p.projectUid)

        _, fs = self._fetch_and_format(req, p, self.long_feature_format)

        if not fs:
            raise gws.web.error.NotFound()

        return FsDetailsResponse({
            'feature': fs[0],
        })

    def api_fs_export(self, req: t.IRequest, p: FsExportParams) -> FsExportResponse:
        """Export Flurstueck features"""

        self._precheck_request(req, p.projectUid)

        _, features = self._fetch(req, p)

        if not features:
            raise gws.web.error.NotFound()

        job_uid = gws.random_string(64)
        out_path = f'{gws.TMP_DIR}/{job_uid}fs.export.csv'

        export.as_csv(self, (f.attributes for f in features), p.groups, out_path)

        job = gws.tools.job.create(
            uid=job_uid,
            user_uid=req.user.full_uid,
            worker='',
            args='')

        job.update(gws.tools.job.State.complete, result=out_path)

        return FsExportResponse({
            'url': gws.SERVER_ENDPOINT + '?cmd=assetHttpGetResult&jobUid=' + job_uid,
        })

    def api_fs_print(self, req: t.IRequest, p: FsPrintParams) -> gws.common.printer.types.PrinterResponse:
        """Print Flurstueck features"""

        self._precheck_request(req, p.projectUid)

        _, features = self._fetch(req, p)

        if not features:
            raise gws.web.error.NotFound()

        pp = p.printParams
        pp.projectUid = p.projectUid
        pp.locale = p.locale
        pp.templateUid = self.print_template.uid
        pp.sections = []

        for feature in features:
            center = feature.shape.geo.centroid
            pp.sections.append(gws.common.printer.types.PrintSection({
                'center': [center.x, center.y],
                'data': {
                    'feature': feature,
                    'attributes': feature.attributes,
                },
                'items': [
                    gws.common.printer.types.PrintItem({
                        'features': [feature.props],
                        'style': p.highlightStyle,
                    })
                ]
            }))

        return gws.common.printer.service.start_job(req, pp)

    def api_geocoder(self, req: t.IRequest, p: GeocoderParams) -> GeocoderResponse:

        # NB don't check disableApi here
        if not self.has_index:
            raise gws.web.error.NotFound()

        target_crs = p.crs
        cs = []

        for ad in p.adressen:
            q = {k: ad.get(k) for k in _GEOCODER_ADDR_KEYS if ad.get(k)}

            if not q:
                cs.append(None)
                continue

            total, features = self._find_address(FsAddressQueryParams(q), target_crs, limit=1)

            if not total:
                cs.append(None)
                continue

            for feature in features:
                cs.append([
                    round(feature.shape.geo.x, 2),
                    round(feature.shape.geo.y, 2)
                ])
                break

        return GeocoderResponse({
            'coordinates': cs
        })

    def index_create(self, user, password):
        with self._connect_for_write(user, password) as conn:
            index.create(conn, read_user=self.connect_args['params']['user'])

    def index_drop(self, user, password):
        with self._connect_for_write(user, password) as conn:
            index.drop(conn)

    def index_ok(self):
        with self._connect() as conn:
            return index.ok(conn)

    def find_fs(self, query: FsQueryParams, target_crs, allow_eigentuemer=False, allow_buchung=False, limit=None):
        # public api method for other ext's

        return self._find(query, target_crs, allow_eigentuemer, allow_buchung, limit)

    def find_address(self, query: FsAddressQueryParams, target_crs, limit=None):
        # public api method for other ext's

        return self._find_address(query, target_crs, limit)

    def _precheck_request(self, req: t.IRequest, project_uid):
        if self.disableApi:
            raise gws.web.error.Forbidden()
        req.require_project(project_uid)
        if not self.has_index:
            raise gws.web.error.NotFound()

    def _set_gemeinde_or_gemarkung(self, query):
        uid = query.get('gemarkungOrGemeindeUid')
        if uid:
            del query.gemarkungOrGemeindeUid
            uid = uid.split(':')
            if len(uid) == 2 and uid[0] == 'gemeinde':
                query.gemeindeUid = uid[1]
            if len(uid) == 2 and uid[0] == 'gemarkung':
                query.gemarkungUid = uid[1]

    def _fetch_and_format(self, req, query: FsQueryParams, fmt: t.IFormat):
        fprops = []
        total, features = self._fetch(req, query)

        for feature in features:
            feature.apply_format(fmt)
            props = feature.props
            del props.attributes
            fprops.append(props)

        return total, sorted(fprops, key=lambda p: p.title)

    def _fetch(self, req, query: FsQueryParams):
        project = req.require_project(query.projectUid)
        target_crs = project.map.var('crs')

        eigentuemer_flag = self._eigentuemer_flag(req, query)
        if eigentuemer_flag < 0:
            self._log_eigentuemer_access(req, query, check=False)
            raise gws.web.error.BadRequest()

        allow_eigentuemer = eigentuemer_flag > 0
        allow_buchung = self._can_read_buchung(req)

        if query.get('shapes'):
            shape = gws.gis.shape.union(gws.gis.shape.from_props(s) for s in query.get('shapes'))
            if shape:
                query.shape = shape

        total, features = self._find(query, target_crs, allow_eigentuemer, allow_buchung, self.limit)
        for f in features:
            f.attributes['is_guest_user'] = req.user.is_guest

        gws.log.debug(
            f'FS_SEARCH eigentuemer_flag={eigentuemer_flag} query={query!r} total={total!r} len={len(features)}')

        if allow_eigentuemer:
            self._log_eigentuemer_access(req, query, check=True, total=total, features=features)

        return total, features

    def _find(self, query: FsQueryParams, target_crs, allow_eigentuemer, allow_buchung, limit):
        features = []

        query = self._expand_combined_params(query)
        query = self._remove_restricted_params(query, allow_eigentuemer, allow_buchung)

        with self._connect() as conn:
            total, rs = flurstueck.find(conn, query, limit)
            for rec in rs:
                rec = self._remove_restricted_data(rec, allow_eigentuemer, allow_buchung)
                feature = gws.gis.feature.new({
                    'uid': rec['gml_id'],
                    'attributes': rec,
                    'shape': gws.gis.shape.from_wkb(rec['geom'], self.crs)
                })
                features.append(feature.transform(target_crs))

        return total, features

    def _find_address(self, query: FsAddressQueryParams, target_crs, limit):
        features = []

        query = self._expand_combined_params(query)

        with self._connect() as conn:
            total, rs = adresse.find(conn, query, limit)
            for rec in rs:
                feature = gws.gis.feature.new({
                    'uid': rec['gml_id'],
                    'attributes': rec,
                    'shape': gws.gis.shape.from_xy(rec['x'], rec['y'], self.crs)
                })
                features.append(feature.transform(target_crs))

        return total, features

    def _eigentuemer_flag(self, req: t.IRequest, p: FsQueryParams):
        if not self._can_read_eigentuemer(req):
            return 0
        if not self.control_mode:
            return 1
        if not p.wantEigentuemer:
            return 0

        r = self._check_control_input(p.controlInput)
        gws.log.debug(f'controlInput={p.controlInput!r} result={r!r}')
        if not r:
            return -1

        return 1

    _log_eigentuemer_access_params = ['fsUids', 'bblatt', 'vorname', 'name']

    def _log_eigentuemer_access(self, req: t.IRequest, p: FsQueryParams, check, total=0, features=None):
        if not self.log_table:
            gws.log.debug('_log_eigentuemer_access', check, 'no log table!')
            return

        has_relevant_params = any(p.get(s) for s in self._log_eigentuemer_access_params)

        if check and not has_relevant_params:
            gws.log.debug('_log_eigentuemer_access', check, 'no relevant params!')
            return

        fs_ids = ''
        if features:
            fs_ids = ','.join(f.uid for f in features)

        data = {
            'app_name': 'gws',
            'date_time': gws.tools.date.now_iso(),
            'ip': req.env('REMOTE_ADDR', ''),
            'login': req.user.uid,
            'user_name': req.user.display_name,
            'control_input': (p.controlInput or '').strip(),
            'control_result': 1 if check else 0,
            'fs_count': total,
            'fs_ids': fs_ids
        }

        with self._connect() as conn:
            conn.insert_one(self.log_table, 'id', data)

        gws.log.debug('_log_eigentuemer_access', check, 'ok')

    def _expand_combined_params(self, query):
        s = gws.get(query, 'alkisFs')
        if s:
            self._expand_combined_params2(query, s, _COMBINED_FS_PARAMS)

        s = gws.get(query, 'alkisAd')
        if s:
            self._expand_combined_params2(query, s, _COMBINED_AD_PARAMS)

        return query

    def _expand_combined_params2(self, query: FsQueryParams, value, fields):
        vs = value.split(_COMBINED_PARAMS_DELIM)

        for field in fields:
            if vs and vs[0] != '0':
                setattr(query, field, vs[0])
            vs = vs[1:]

    def _remove_restricted_params(self, query: FsQueryParams, allow_eigentuemer, allow_buchung):
        if not allow_eigentuemer:
            gws.popattr(query, 'vorname')
            gws.popattr(query, 'name')
        if not allow_buchung:
            gws.popattr(query, 'bblatt')
        return query

    def _remove_restricted_data(self, rec, allow_eigentuemer, allow_buchung):
        if allow_eigentuemer:
            return rec

        if allow_buchung:
            for b in rec.get('buchung', []):
                b.pop('eigentuemer', None)
            return rec

        rec.pop('buchung', None)
        return rec

    def _check_control_input(self, inp):
        if not self.control_rules:
            return True

        inp = (inp or '').strip()

        for rule in self.control_rules:
            if re.search(rule, inp):
                return True

        return False

    def _can_read_eigentuemer(self, req):
        b = req.user.can_use(self.eigentuemer, parent=self)
        gws.log.debug(f'_can_read_eigentuemer user={req.user.full_uid!r} res={b}')
        return b

    def _can_read_buchung(self, req):
        b = req.user.can_use(self.buchung, parent=self)
        gws.log.debug(f'_can_read_buchung user={req.user.full_uid!r} res={b}')
        return b

    def _connect(self):
        return connection.AlkisConnection(**self.connect_args)

    def _connect_for_write(self, user, password):
        params = gws.extend(self.connect_args['params'], {
            'user': user,
            'password': password,
        })
        connect_args = gws.extend(self.connect_args, {'params': params})
        return connection.AlkisConnection(**connect_args)

    def _get_alkis_crs(self, db):
        with db.connect() as conn:
            return conn.crs_for_column(
                self.var('alkisSchema') + '.ax_flurstueck',
                'wkb_geometry')
