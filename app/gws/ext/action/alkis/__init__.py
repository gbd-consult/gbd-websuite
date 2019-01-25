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
import gws.types as t
import gws.tools.json2
import gws.web
from .data import index, flurstueck
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
    """access to the Eigentümer (eigentuemer) information"""

    access: t.Access  #: access rights

    controlMode: bool = False  #: restricted mode enabled
    controlRules: t.Optional[t.List[str]]  #: list of regular expression for the restricted input control

    logTable: str = ''  #: data access protocol table name


class BuchungConfig:
    """access to the Grundbuch (register) information"""

    access: t.Access  #: access rights


class Config(t.WithTypeAndAccess):
    """Flurstückssuche (cadaster parlcels search) action"""

    db: t.Optional[str]  #: database (postgis) provider ID
    dataSchema: str = 'public'  #: schema where ALKIS tables are stored, must be readable
    indexSchema: str = 'gws'  #: schema to store gws internal indexes, must be writable

    eigentuemer: t.Optional[EigentuemerConfig]  #: access to the Eigentümer (eigentuemer) information
    buchung: t.Optional[BuchungConfig]  #: access to the Grundbuch (register) information

    excludeGemarkung: t.Optional[t.List[str]]  #: Gemarkung (AU) IDs to exclude from search

    limit: int = 100  #: search results limit

    featureFormat: t.Optional[t.FormatConfig]  #: template for on-screen Flurstueck details
    printTemplate: t.Optional[t.ext.template.Config]  #: template for printed Flurstueck details

    export: bool = False  #: export function enabled
    select: bool = False  #: select mode enabled


class Gemarkung(t.Data):
    """Gemarkung (Administative Unit) object"""

    name: str  #: name
    uid: str  #: unique ID


class FsSetupParams(t.Data):
    projectUid: str


class FsSetupResponse(t.Response):
    withEigentuemer: bool
    withBuchung: bool
    withExport: bool
    withSelect: bool
    withControl: bool
    withFlurnummer: bool
    gemarkungen: t.List[Gemarkung]
    printTemplate: t.TemplateProps


class FsStrassenParams(t.Data):
    projectUid: str
    gemarkungUid: str


class FsStrassenResponse(t.Response):
    strassen: t.List[str]


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
    shape: t.Optional[t.ShapeProps]


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


_cwd = os.path.dirname(__file__)

DEFAULT_FORMAT = t.FormatConfig({
    'teaser': t.TemplateConfig({
        'type': 'html',
        'path': _cwd + '/templates/feature_teaser.cx.html'
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
            'data_schema': self.var('dataSchema'),
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

        fmt = self.var('featureFormat', default=DEFAULT_FORMAT)

        self.short_feature_format = self.add_child('gws.common.format', t.Config({
            'title': fmt.get('title'),
            'teaser': fmt.get('teaser')
        }))

        self.long_feature_format = self.add_child('gws.common.format', t.Config({
            'title': fmt.get('title'),
            'teaser': fmt.get('teaser'),
            'description': fmt.get('description'),
        }))

        self.print_template: t.TemplateObject = self.add_child(
            'gws.ext.template',
            self.var('printTemplate', default=DEFAULT_PRINT_TEMPLATE))

        # self.export_template: t.TemplateObject = None
        # if self.var('exportTemplate'):
        #     self.export_template = self.add_child(
        #         'gws.ext.template',
        #         self.var('exportTemplate'))

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
                'withExport': self.var('export'),
                'withSelect': self.var('select'),
                'withFlurnummer': flurstueck.has_flurnummer(conn),
                'gemarkungen': flurstueck.gemarkung_list(conn),
                'printTemplate': self.print_template.props,
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

        # self.export_template.render(context={
        #     'features': features
        # }, out_path=out_path)
        #

        with open(out_path, 'w') as fp:
            export.as_csv((f.attributes for f in features), p.groups, fp)

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

    def index_create(self, user, password):
        with self._connect_for_write(user, password) as conn:
            index.create(conn, read_user=self.connect_args['params']['user'])

    def index_drop(self, user, password):
        with self._connect_for_write(user, password) as conn:
            index.drop(conn)

    def index_ok(self):
        with self._connect() as conn:
            return index.ok(conn)

    def _precheck_request(self, req, project_uid):
        req.require_project(project_uid)
        if not self.has_index:
            raise gws.web.error.NotFound()

    def _fetch_and_format(self, req, p: FsQueryParams, fmt: t.FormatInterface):
        fprops = []
        total, features = self._fetch(req, p)

        for feature in features:
            feature.apply_format(fmt)
            props = feature.props
            del props.attributes
            fprops.append(props)

        return total, sorted(fprops, key=lambda p: p.title)

    def _fetch(self, req, p: FsQueryParams):
        project = req.require_project(p.projectUid)
        target_crs = project.map.var('crs')

        eigentuemer_flag = self._eigentuemer_flag(req, p)
        if eigentuemer_flag < 0:
            self._log_eigentuemer_access(req, p, check=False)
            raise gws.web.error.BadRequest()

        allow_eigentuemer = eigentuemer_flag > 0
        allow_buchung = self._can_read_buchung(req)

        p = self._remove_restricted_params(p, allow_eigentuemer, allow_buchung)

        if p.get('shape'):
            p.shape = gws.gis.shape.from_props(p.shape)

        features = []

        with self._connect() as conn:
            total, rs = flurstueck.find(conn, p, self.var('limit'))
            for rec in rs:
                rec = self._remove_restricted_data(rec, allow_eigentuemer, allow_buchung)
                feature = gws.gis.feature.Feature({
                    'uid': rec['gml_id'],
                    'attributes': rec,
                    'shape': gws.gis.shape.from_wkb(rec['geom'], self.crs)
                })
                features.append(feature.transform(target_crs))

        gws.log.debug(f'FS_SEARCH eigentuemer_flag={eigentuemer_flag} total={total!r} len={len(features)}')
        gws.p(p)

        if allow_eigentuemer:
            self._log_eigentuemer_access(req, p, check=True, total=total, features=features)

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


    def _remove_restricted_params(self, p: FsQueryParams, allow_eigentuemer, allow_buchung):
        if not allow_eigentuemer:
            del p.vorname
            del p.name
        if not allow_buchung:
            del p.bblatt
        return p

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
        b = req.user.can_read(self.eigentuemer)
        gws.log.debug(f'_can_read_eigentuemer user={req.user.full_uid!r} res={b}')
        return b

    def _can_read_buchung(self, req):
        b = req.user.can_read(self.buchung)
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
                self.var('dataSchema') + '.ax_flurstueck',
                'wkb_geometry')
