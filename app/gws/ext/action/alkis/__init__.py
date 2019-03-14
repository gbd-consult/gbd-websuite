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
import gws.tools.misc
import gws.tools.storage
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
    fs_ids varchar(255)
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
    """UI configuration."""

    export: bool = False  #: export function enabled
    select: bool = False  #: select mode enabled
    pick: bool = False  #: pick mode enabled
    searchSelection: bool = False  #: search in selection enabled
    searchSpatial: bool = False  #: spatial search enabled


class Config(t.WithTypeAndAccess):
    """Flurstückssuche (cadaster parlcels search) action"""

    db: t.Optional[str]  #: database provider ID
    alkisSchema: str = 'public'  #: schema where ALKIS tables are stored, must be readable
    indexSchema: str = 'gws'  #: schema to store gws internal indexes, must be writable

    eigentuemer: t.Optional[EigentuemerConfig]  #: access to the Eigentümer (owner) information
    buchung: t.Optional[BuchungConfig]  #: access to the Grundbuch (register) information

    excludeGemarkung: t.Optional[t.List[str]]  #: Gemarkung (AU) IDs to exclude from search

    limit: int = 100  #: search results limit

    featureFormat: t.Optional[t.FormatConfig]  #: template for on-screen Flurstueck details
    printTemplate: t.Optional[t.ext.template.Config]  #: template for printed Flurstueck details

    ui: t.Optional[UiConfig] #: ui options


class Gemarkung(t.Data):
    """Gemarkung (Administative Unit) object"""

    name: str  #: name
    uid: str  #: unique ID


class FsSetupParams(t.Data):
    projectUid: str


class FsSetupResponse(t.Response):
    withEigentuemer: bool
    withBuchung: bool
    withControl: bool
    withFlurnummer: bool
    gemarkungen: t.List[Gemarkung]
    printTemplate: t.TemplateProps
    limit: int

    uiExport: bool
    uiSelect: bool
    uiPick: bool
    uiSearchSelection: bool
    uiSearchSpatial: bool


class FsStrassenParams(t.Data):
    projectUid: str
    gemarkungUid: str


class FsStrassenResponse(t.Response):
    strassen: t.List[str]


_COMBINED_FS_PARAMS = ['landUid', 'gemarkungUid', 'flurnummer', 'zaehler', 'nenner', 'flurstuecksfolge']
_COMBINED_AD_PARAMS = ['strasse', 'hausnummer', 'plz', 'gemeinde', 'bisHausnummer']

_COMBINED_PARAMS_DELIM = '_'


class FsQueryParams(t.Data):
    projectUid: str
    wantEigentuemer: t.Optional[bool]
    controlInput: t.Optional[str]
    fsUids: t.Optional[t.List[str]]

    bblatt: t.Optional[str]
    flaecheBis: t.Optional[str]
    flaecheVon: t.Optional[str]
    gemarkungUid: t.Optional[str]
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


class FsAddressQueryParams(t.Data):
    land: t.Optional[str]
    regierungsbezirk: t.Optional[str]
    kreis: t.Optional[str]
    gemeinde: t.Optional[str]
    gemarkung: t.Optional[str]
    landUid: t.Optional[str]
    regierungsbezirkUid: t.Optional[str]
    kreisUid: t.Optional[str]
    gemeindeUid: t.Optional[str]
    gemarkungUid: t.Optional[str]
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


class FsSaveSelectionParams(t.Data):
    projectUid: str
    name: str
    fsUids: t.List[str]


class FsSaveSelectionResponse(t.Response):
    names: t.List[str]


class FsLoadSelectionParams(t.Data):
    projectUid: str
    name: str


class FsLoadSelectionResponse(t.Response):
    features: t.List[t.FeatureProps]


class FsGetSaveNamesParams(t.Data):
    projectUid: str


class FsGetSaveNamesResponse(t.Response):
    names: t.List[str]


_cwd = os.path.dirname(__file__)

DEFAULT_FORMAT = t.FormatConfig({
    'title': t.TemplateConfig({
        'type': 'html',
        'text': '{attributes.vollnummer}'
    }),
    'teaser': t.TemplateConfig({
        'type': 'html',
        'text': 'Flurstück {attributes.vollnummer}'
    }),
    'description': t.TemplateConfig({
        'type': 'html',
        'path': _cwd + '/templates/data.cx.html'
    })
})

DEFAULT_PRINT_TEMPLATE = t.TemplateConfig({
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
    def configure(self):
        super().configure()

        prov_uid = self.var('db')
        if prov_uid:
            prov = self.root.find('gws.ext.db.provider', prov_uid)
        else:
            prov = self.root.find_first('gws.ext.db.provider')

        self.crs = self._get_alkis_crs(prov)
        self.has_index = False

        self.connect_args = {
            'params': prov.connect_params,
            'index_schema': self.var('indexSchema'),
            'data_schema': self.var('alkisSchema'),
            'crs': self.crs,
            'exclude_gemarkung': self.var('excludeGemarkung')
        }

        if self.index_ok():
            gws.log.info(f'ALKIS indexes in "{prov.uid}" are fine')
            self.has_index = True
        else:
            gws.log.warn(f'ALKIS indexes in "{prov.uid}" are not ok, please reindex')

        if not self.has_index:
            return

        self.limit = int(self.var('limit'))

        fmt = self.var('featureFormat') or t.FormatConfig()
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

        self.print_template: t.TemplateObject = self.add_child(
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

    def api_fs_setup(self, req, p: FsSetupParams) -> FsSetupResponse:
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
                'uiExport': self.var('ui.export'),
                'uiSelect': self.var('ui.select'),
                'uiPick': self.var('ui.pick'),
                'uiSearchSelection': self.var('ui.searchSelection'),
                'uiSearchSpatial': self.var('ui.searchSpatial'),
            })

    def api_fs_strassen(self, req, p: FsStrassenParams) -> FsStrassenResponse:
        """Return a list of Strassen for the given Gemarkung"""

        self._precheck_request(req, p.projectUid)

        with self._connect() as conn:
            return FsStrassenResponse({
                'strassen': flurstueck.strasse_list(conn, p.gemarkungUid),
            })

    def api_fs_search(self, req, p: FsQueryParams) -> FsSearchResponse:
        """Perform a Flurstueck search"""

        self._precheck_request(req, p.projectUid)

        total, fs = self._fetch_and_format(req, p, self.short_feature_format)

        if total > self.limit:
            raise gws.web.error.Conflict()

        return FsSearchResponse({
            'total': total,
            'features': sorted(fs, key=lambda p: p.title)
        })

    def api_fs_details(self, req, p: FsDetailsParams) -> FsDetailsResponse:
        """Return a Flurstueck feature with details"""

        self._precheck_request(req, p.projectUid)

        _, fs = self._fetch_and_format(req, p, self.long_feature_format)

        if not fs:
            raise gws.web.error.NotFound()

        return FsDetailsResponse({
            'feature': fs[0],
        })

    def api_fs_export(self, req, p: FsExportParams) -> FsExportResponse:
        """Export Flurstueck features"""

        self._precheck_request(req, p.projectUid)

        if not self.var('export'):
            raise gws.web.error.NotFound()

        _, features = self._fetch(req, p)

        if not features:
            raise gws.web.error.NotFound()

        job_uid = gws.random_string(64)
        out_path = '/tmp/' + job_uid + 'fs.export.csv'

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

    def api_fs_print(self, req, p: FsPrintParams) -> gws.common.printer.types.PrinterResponse:
        """Print Flurstueck features"""

        self._precheck_request(req, p.projectUid)

        _, features = self._fetch(req, p)

        if not features:
            raise gws.web.error.NotFound()

        pp = p.printParams
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

    def api_fs_save_selection(self, req, p: FsSaveSelectionParams) -> FsSaveSelectionResponse:
        self._precheck_request(req, p.projectUid)

        gws.tools.storage.put('alkis', p.name, req.user.full_uid, p.fsUids)
        names = gws.tools.storage.get_names('alkis', req.user.full_uid)
        return FsSaveSelectionResponse({
            'names': names
        })

    def api_fs_load_selection(self, req, p: FsLoadSelectionParams) -> FsLoadSelectionResponse:
        self._precheck_request(req, p.projectUid)

        query = FsQueryParams({
            'projectUid': p.projectUid,
            'fsUids': gws.tools.storage.get('alkis', p.name, req.user.full_uid)
        })

        total, fs = self._fetch_and_format(req, query, self.short_feature_format)

        return FsLoadSelectionResponse({
            'features': sorted(fs, key=lambda p: p.title)
        })

    def api_fs_get_save_names(self, req, p: FsGetSaveNamesParams) -> FsGetSaveNamesResponse:
        names = gws.tools.storage.get_names('alkis', req.user.full_uid)
        return FsGetSaveNamesResponse({
            'names': names
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

    def _precheck_request(self, req, project_uid):
        req.require_project(project_uid)
        if not self.has_index:
            raise gws.web.error.NotFound()

    def _fetch_and_format(self, req, query: FsQueryParams, fmt: t.FormatInterface):
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

        gws.log.debug(f'FS_SEARCH eigentuemer_flag={eigentuemer_flag} total={total!r} len={len(features)}')
        gws.p(query)

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
                feature = gws.gis.feature.Feature({
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
                feature = gws.gis.feature.Feature({
                    'uid': rec['gml_id'],
                    'attributes': rec,
                    'shape': gws.gis.shape.from_xy(rec['x'], rec['y'], self.crs)
                })
                features.append(feature.transform(target_crs))

        return total, features

    def _eigentuemer_flag(self, req, p: FsQueryParams):
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

    def _log_eigentuemer_access(self, req, p: FsQueryParams, check, total=0, features=None):
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
            'ip': req.environ.get('REMOTE_ADDR', ''),
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

    def _get_alkis_crs(self, prov):
        with prov.connect() as conn:
            return conn.crs_for_column(
                self.var('alkisSchema') + '.ax_flurstueck',
                'wkb_geometry')
