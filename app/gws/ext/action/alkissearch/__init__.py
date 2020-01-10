"""Backend for the Flurstückssuche (cadaster parlcels search) form."""

import os
import re

import gws
import gws.tools.date
import gws.tools.job
import gws.gis.shape
import gws.common.printer.service
import gws.common.printer.types
import gws.common.template
import gws.ext.db.provider.postgres
import gws.web

import gws.ext.tool.alkis as alkis

import gws.types as t

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


class EigentuemerConfig(t.WithAccess):
    """Access to the Eigentümer (owner) information"""

    controlMode: bool = False  #: restricted mode enabled
    controlRules: t.Optional[t.List[str]]  #: list of regular expression for the restricted input control

    logTable: str = ''  #: data access protocol table name


class BuchungConfig(t.WithAccess):
    """Access to the Grundbuch (register) information"""
    pass


class UiConfig(t.Config):
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

    eigentuemer: t.Optional[EigentuemerConfig]  #: access to the Eigentümer (owner) information
    buchung: t.Optional[BuchungConfig]  #: access to the Grundbuch (register) information
    limit: int = 100  #: search results limit
    featureFormat: t.Optional[gws.common.template.FeatureFormatConfig]  #: template for on-screen Flurstueck details
    printTemplate: t.Optional[t.ext.template.Config]  #: template for printed Flurstueck details
    ui: t.Optional[UiConfig]  #: ui options


class Props(t.Props):
    type: t.Literal = 'alkissearch'
    withEigentuemer: bool
    withBuchung: bool
    withControl: bool
    withFlurnummer: bool
    gemarkungen: t.List[alkis.Gemarkung]
    printTemplate: t.TemplateProps
    limit: int
    ui: UiConfig


##

class FindParams(t.Params):
    gemarkung: str = ''
    gemarkungOrGemeindeUid: str = ''
    gemarkungUid: str = ''
    gemeinde: str = ''
    gemeindeUid: str = ''


class FindStrasseParams(FindParams):
    pass


class FindStrasseResponse(t.Response):
    strassen: t.List[str]


class FindFlurstueckParams(FindParams):
    wantEigentuemer: t.Optional[bool]
    controlInput: t.Optional[str]
    crs: t.Optional[t.Crs]
    shapes: t.Optional[t.List[t.ShapeProps]]

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


class FindFlurstueckResponse(t.Response):
    features: t.List[t.FeatureProps]
    total: int


class FindAdresseParams(FindParams):
    crs: t.Optional[t.Crs]

    bisHausnummer: str = ''
    hausnummer: str = ''
    kreis: str = ''
    kreisUid: str = ''
    land: str = ''
    landUid: str = ''
    regierungsbezirk: str = ''
    regierungsbezirkUid: str = ''
    strasse: str = ''


class FindAdresseResponse(t.Response):
    features: t.List[t.FeatureProps]
    total: int


class GetDetailsParams(FindFlurstueckParams):
    pass


class GetDetailsResponse(t.Response):
    feature: t.FeatureProps


class PrintParams(t.Params):
    findParams: FindFlurstueckParams
    printParams: gws.common.printer.types.PrintParamsWithTemplate
    highlightStyle: t.StyleProps


class ExportParams(FindFlurstueckParams):
    groups: t.List[str]


class ExportResponse(t.Response):
    url: str


##


_dir = os.path.dirname(__file__)

DEFAULT_FORMAT = gws.common.template.FeatureFormatConfig(
    title=gws.common.template.Config(type='html', text='{vollnummer}'),
    teaser=gws.common.template.Config(type='html', text='Flurstück {vollnummer}'),
    description=gws.common.template.Config(type='html', path=f'{_dir}/templates/data.cx.html')
)

DEFAULT_PRINT_TEMPLATE = gws.common.template.Config(
    type='html',
    path=f'{_dir}/templates/print.cx.html',
    pageWidth=210,
    pageHeight=297,
    mapWidth=100,
    mapHeight=100,
    qualityLevels=[t.TemplateQualityLevel(name='default', dpi=150)]
)

_EF_DENY = 0  # no access to Eigentümer
_EF_ALLOW = 1  # granted access to Eigentümer
_EF_FAIL = -1  # access to Eigentümer granted, control check failed


##


class Object(gws.ActionObject):
    def __init__(self):
        super().__init__()

        self.alkis: alkis.Object = None
        self.valid = False

        self.buchung: BuchungConfig = None
        self.control_mode = False
        self.control_rules = []
        self.eigentuemer: EigentuemerConfig = None
        self.limit = 0
        self.log_table = ''
        self.long_feature_format: t.IFormat = None
        self.print_template: t.ITemplate = None
        self.short_feature_format: t.IFormat = None

    def configure(self):
        super().configure()

        self.alkis: gws.ext.tool.alkis.Object = self.find_first('gws.ext.tool.alkis')
        if not self.alkis or not self.alkis.has_index:
            gws.log.warn('alkissearch cannot init, no alkis index found')
            return

        self.valid = True
        self.limit = int(self.var('limit'))

        fmt = self.var('featureFormat') or gws.common.template.FeatureFormatConfig()
        for f in 'title', 'teaser', 'description':
            if not fmt.get(f):
                setattr(fmt, f, DEFAULT_FORMAT.get(f))

        self.short_feature_format = self.add_child('gws.common.format', t.Config(
            title=fmt.title,
            teaser=fmt.teaser,
        ))

        self.long_feature_format = self.add_child('gws.common.format', t.Config(
            title=fmt.title,
            teaser=fmt.teaser,
            description=fmt.description,
        ))

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
            with self.alkis.db.connect() as conn:
                if not conn.user_can('INSERT', self.log_table):
                    raise ValueError(f'no INSERT acccess to {self.log_table!r}')

    def props_for(self, user):
        if not self.valid:
            return None
        return {
            'type': self.type,
            'withEigentuemer': self._can_read_eigentuemer(user),
            'withControl': self._can_read_eigentuemer(user) and self.control_mode,
            'withBuchung': self._can_read_buchung(user),
            'withFlurnummer': self.alkis.has_flurnummer,
            'gemarkungen': self.alkis.gemarkung_list(),
            'printTemplate': self.print_template.props,
            'limit': self.limit,
            'ui': self.var('ui'),
        }

    def api_find_strasse(self, req: t.IRequest, p: FindStrasseParams) -> FindStrasseResponse:
        """Return a list of Strassen for the given Gemarkung"""

        self._validate_request(req, p)
        return FindStrasseResponse(self.alkis.find_strasse(alkis.FindStrasseQuery(p)))

    def api_find_flurstueck(self, req: t.IRequest, p: FindFlurstueckParams) -> FindFlurstueckResponse:
        """Perform a Flurstueck search"""

        self._validate_request(req, p)
        return self._fetch_and_format(req, p, self.short_feature_format, self.limit, self.limit)

    def api_get_details(self, req: t.IRequest, p: GetDetailsParams) -> GetDetailsResponse:
        """Return a Flurstueck feature with details"""

        self._validate_request(req, p)
        res = self._fetch_and_format(req, p, self.long_feature_format, 1, self.limit)

        if not res.features:
            raise gws.web.error.NotFound()

        return GetDetailsResponse(feature=res.features[0])

    def api_export(self, req: t.IRequest, p: ExportParams) -> ExportResponse:
        """Export Flurstueck features"""

        self._validate_request(req, p)
        res = self._fetch(req, p)

        if not res.features:
            raise gws.web.error.NotFound()

        job_uid = gws.random_string(64)
        out_path = f'{gws.TMP_DIR}/{job_uid}fs.export.csv'

        # FILE RESPONSE

        # export.as_csv(self, (f.attributes for f in features), p.groups, out_path)

        job = gws.tools.job.create(job_uid, req.user, worker='')
        job.update(gws.tools.job.State.complete, result=out_path)

        # return t.FileResponse({
        #     'url': gws.SERVER_ENDPOINT + '?cmd=assetHttpGetResult&jobUid=' + job_uid,
        # })

    def api_print(self, req: t.IRequest, p: PrintParams) -> gws.common.printer.types.PrinterResponse:
        """Print Flurstueck features"""

        self._validate_request(req, p)

        fp = p.findParams
        fp.projectUid = p.projectUid
        fp.locale = p.locale

        res = self._fetch(req, fp)

        if not res.features:
            raise gws.web.error.NotFound()

        pp = p.printParams
        pp.projectUid = p.projectUid
        pp.locale = p.locale
        pp.templateUid = self.print_template.uid
        pp.sections = []

        for feature in res.features:
            center = feature.shape.centroid
            pp.sections.append(gws.common.printer.types.PrintSection(
                center=[center.x, center.y],
                context=feature.template_context,
                items=[
                    gws.common.printer.types.PrintItemFeatures(
                        type='features',
                        features=[feature.props],
                        style=p.highlightStyle,
                    )
                ]
            ))

        return gws.common.printer.service.start_job(req, pp)

    ##

    def _validate_request(self, req: t.IRequest, p: t.Params):
        if not self.valid:
            raise gws.web.error.NotFound()
        req.require_project(p.projectUid)

    def _fetch_and_format(self, req, p: FindFlurstueckParams, fmt: t.IFormat, soft_limit, hard_limit) -> FindFlurstueckResponse:
        fprops = []
        res = self._fetch(req, p, soft_limit, hard_limit)

        for f in res.features:
            f.apply_format(fmt)
            props = f.props
            del props.attributes
            fprops.append(props)

        return FindFlurstueckResponse(
            total=res.total,
            features=sorted(fprops, key=lambda f: f.elements['title']))

    def _fetch(self, req, p: FindFlurstueckParams, soft_limit=0, hard_limit=0) -> alkis.FindFlurstueckResult:

        fq = alkis.FindFlurstueckQuery(p)

        eigentuemer_flag = self._eigentuemer_flag(req, p)
        if eigentuemer_flag == _EF_FAIL:
            self._log_eigentuemer_access(req, p, is_ok=False)
            raise gws.web.error.BadRequest()

        fq.withEigentuemer = eigentuemer_flag == _EF_ALLOW
        fq.withBuchung = self._can_read_buchung(req.user)

        if p.get('shapes'):
            shape = gws.gis.shape.union(gws.gis.shape.from_props(s) for s in p.get('shapes'))
            if shape:
                fq.shape = shape.transformed(self.alkis.crs)

        if soft_limit:
            fq.limit = soft_limit

        res = self.alkis.find_flurstueck(fq)

        gws.log.debug(f'FS_SEARCH ef={eigentuemer_flag} query={p!r} total={res.total!r} len={len(res.features)}')

        if hard_limit and res.total > hard_limit:
            raise gws.web.error.Conflict()

        project = req.require_project(p.projectUid)
        crs = p.get('crs') or project.map.crs

        for f in res.features:
            f.transform(crs)
            f.attributes.append(t.Attribute(name='is_guest_user', value=req.user.is_guest))

        if fq.withEigentuemer:
            self._log_eigentuemer_access(req, p, is_ok=True, total=res.total, features=res.features)

        return res

    def _eigentuemer_flag(self, req: t.IRequest, p: FindFlurstueckParams):
        if not self._can_read_eigentuemer(req.user):
            return _EF_DENY
        if not self.control_mode:
            return _EF_ALLOW
        if not p.wantEigentuemer:
            return _EF_DENY

        r = self._check_control_input(p.controlInput)
        gws.log.debug(f'controlInput={p.controlInput!r} result={r!r}')
        if not r:
            return _EF_FAIL

        return _EF_ALLOW

    _log_eigentuemer_access_params = ['fsUids', 'bblatt', 'vorname', 'name']

    def _log_eigentuemer_access(self, req: t.IRequest, p: FindFlurstueckParams, is_ok, total=0, features=None):
        if not self.log_table:
            gws.log.debug('_log_eigentuemer_access', is_ok, 'no log table!')
            return

        has_relevant_params = any(p.get(s) for s in self._log_eigentuemer_access_params)

        if is_ok and not has_relevant_params:
            gws.log.debug('_log_eigentuemer_access', is_ok, 'no relevant params!')
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
            'control_result': 1 if is_ok else 0,
            'fs_count': total,
            'fs_ids': fs_ids
        }

        with self.alkis.connect() as conn:
            conn.insert_one(self.log_table, 'id', data)

        gws.log.debug('_log_eigentuemer_access', is_ok, 'ok')

    def _check_control_input(self, inp):
        if not self.control_rules:
            return True

        inp = (inp or '').strip()

        for rule in self.control_rules:
            if re.search(rule, inp):
                return True

        return False

    def _can_read_eigentuemer(self, user: t.IUser):
        b = user.can_use(self.eigentuemer, parent=self)
        gws.log.debug(f'_can_read_eigentuemer user={user.full_uid!r} res={b}')
        return b

    def _can_read_buchung(self, user: t.IUser):
        b = user.can_use(self.buchung, parent=self)
        gws.log.debug(f'_can_read_buchung user={user.full_uid!r} res={b}')
        return b
