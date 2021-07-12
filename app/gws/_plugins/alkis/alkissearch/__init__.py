"""Backend for the Flurstückssuche (cadaster parlcels search) form."""

import os
import re

import gws.base.print.job
import gws.base.print.types
import gws.ext.db.provider.postgres
import gws.ext.helper.alkis
import gws.ext.helper.alkis.util.export

import gws
import gws.types as t
import gws.base.api
import gws.base.template
import gws.lib.shape
import gws.lib.date
import gws.lib.job
import gws.base.web.error

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


class EigentuemerConfig(gws.WithAccess):
    """Access to the Eigentümer (owner) information"""

    controlMode: bool = False  #: restricted mode enabled
    controlRules: t.Optional[t.List[str]]  #: regular expression for the restricted input control
    logTable: str = ''  #: data access protocol table name


class BuchungConfig(gws.WithAccess):
    """Access to the Grundbuch (register) information"""
    pass


class UiGemarkungListMode(t.Enum):
    none = 'none'  #: do not show the list
    plain = 'plain'  #: only "gemarkung"
    combined = 'combined'  #: "gemarkung (gemeinde)"
    tree = 'tree'  # a tree with level 1 = "gemeinde" and level 2 = "gemarkung"


class UiStrasseListMode(t.Enum):
    plain = 'plain'  #: just "strasse"
    withGemeinde = 'withGemeinde'  #: "strasse" ("gemeinde")
    withGemarkung = 'withGemarkung'  #: "strasse" ("gemarkung")
    withGemeindeIfRepeated = 'withGemeindeIfRepeated'  #: "strasse" ("gemeinde"), when needed for disambiguation
    withGemarkungIfRepeated = 'withGemarkungIfRepeated'  #: "strasse" ("gemarkung"), when needed for disambiguation


class UiBblattSearchMode(t.Enum):
    start = 'start'  #: search from the beginning
    end = 'end'  #: search from the end
    any = 'any'  #: search anywhere
    exact = 'exact'  #: exact search


class UiStrasseSearchMode(t.Enum):
    start = 'start'  #: search from the beginning
    any = 'any'  #: search anywhere


class UiConfig(gws.Config):
    """Flurstückssuche UI configuration."""

    useExport: bool = False  #: export function enabled
    useSelect: bool = False  #: select mode enabled
    usePick: bool = False  #: pick mode enabled
    searchSelection: bool = False  #: search in selection enabled
    searchSpatial: bool = False  #: spatial search enabled
    gemarkungListMode: UiGemarkungListMode = 'combined'  #: gemarkung list mode
    strasseListMode: UiStrasseListMode = 'plain'  #: strasse list entry format
    strasseSearchMode: UiStrasseSearchMode = 'start'  #: strasse search mode
    autoSpatialSearch: bool = False  #: activate spatial search after submit
    bblattSearchMode: UiBblattSearchMode = 'any'  #: buchungsblatt search mode


class Config(gws.WithAccess):
    """Flurstückssuche (cadaster parlcels search) action"""

    helper: t.Optional[str] #: ALKIS helper uid
    eigentuemer: t.Optional[EigentuemerConfig]  #: access to the Eigentümer (owner) information
    buchung: t.Optional[BuchungConfig]  #: access to the Grundbuch (register) information
    limit: int = 100  #: search results limit
    templates: t.Optional[t.List[gws.ext.template.Config]]  #: templates for on-screen Flurstueck details
    printTemplate: t.Optional[gws.ext.template.Config]  #: template for printed Flurstueck details
    ui: t.Optional[UiConfig]  #: ui options
    export: t.Optional[gws.ext.helper.alkis.util.export.Config]  #: csv export configuration


class Props(gws.Props):
    type: t.Literal = 'alkissearch'
    exportGroups: dict
    gemarkungen: t.List[gws.ext.helper.alkis.Gemarkung]
    limit: int
    printTemplate: gws.base.template.TemplateProps
    ui: UiConfig
    withBuchung: bool
    withControl: bool
    withEigentuemer: bool
    withFlurnummer: bool


##

class BaseFindParams(gws.Params):
    gemarkung: str = ''
    gemarkungUid: str = ''
    gemeinde: str = ''
    gemeindeUid: str = ''
    strasse: str = ''
    strasseMode: t.Optional[gws.ext.helper.alkis.StrasseQueryMode]


class GetToponymsParams(gws.Params):
    pass


class ToponymGemeinde(gws.Data):
    name: str
    uid: str


class ToponymGemarkung(gws.Data):
    name: str
    uid: str
    gemeindeUid: str


class ToponymStrasse(gws.Data):
    name: str
    gemarkungUid: str


class GetToponymsResponse(gws.Response):
    gemeinden: t.List[ToponymGemeinde]
    gemarkungen: t.List[ToponymGemarkung]
    strasseNames: t.List[str]
    strasseGemarkungUids: t.List[str]


class FindFlurstueckParams(BaseFindParams):
    wantEigentuemer: t.Optional[bool]
    controlInput: t.Optional[str]
    crs: t.Optional[gws.Crs]
    shapes: t.Optional[t.List[gws.ShapeProps]]

    bblatt: str = ''
    flaecheBis: str = ''
    flaecheVon: str = ''
    flurnummer: str = ''
    flurstuecksfolge: str = ''
    fsUids: t.List[str] = []
    hausnummer: str = ''
    name: str = ''
    nenner: str = ''
    vnum: str = ''
    vorname: str = ''
    zaehler: str = ''


class FindFlurstueckResponse(gws.Response):
    features: t.List[gws.lib.feature.Props]
    total: int


class FindAdresseParams(BaseFindParams):
    crs: t.Optional[gws.Crs]

    bisHausnummer: str = ''
    hausnummer: str = ''
    kreis: str = ''
    kreisUid: str = ''
    land: str = ''
    landUid: str = ''
    regierungsbezirk: str = ''
    regierungsbezirkUid: str = ''


class FindAdresseResponse(gws.Response):
    features: t.List[gws.lib.feature.Props]
    total: int


class GetDetailsParams(FindFlurstueckParams):
    pass


class GetDetailsResponse(gws.Response):
    feature: gws.lib.feature.Props


class PrintParams(gws.Params):
    findParams: FindFlurstueckParams
    printParams: gws.base.print.types.PrintParamsWithTemplate
    highlightStyle: gws.StyleProps


class ExportParams(gws.Params):
    findParams: FindFlurstueckParams
    groups: t.List[str]


class ExportResponse(gws.Response):
    content: str
    mime: str


##


_dir = os.path.dirname(__file__)

_DEFAULT_FEATURE_TEMPLATES = [
    gws.Config(
        subject='feature.title',
        type='html',
        text='{vollnummer}',
    ),
    gws.Config(
        subject='feature.teaser',
        type='html',
        text='Flurstück {vollnummer}',
    ),
    gws.Config(
        subject='feature.description',
        type='html',
        path=f'{_dir}/templates/data.cx.html',
    ),
]

_DEFAULT_PRINT_TEMPLATE = gws.Config(
    type='html',
    path=f'{_dir}/templates/print.cx.html',
    pageWidth=210,
    pageHeight=297,
    mapWidth=100,
    mapHeight=100,
    qualityLevels=[gws.TemplateQualityLevel(name='default', dpi=150)]
)

_EF_DENY = 0  # no access to Eigentümer
_EF_ALLOW = 1  # granted access to Eigentümer
_EF_FAIL = -1  # access to Eigentümer granted, control check failed


##


class Object(gws.base.api.Action):
    alkis: gws.ext.helper.alkis.Object

    def configure(self):
        

        self.valid: bool = False

        p = self.var('helper')
        if p:
             self.alkis = t.cast(gws.ext.helper.alkis.Object, self.root.find('gws.ext.helper.alkis', p))
        else:
            self.alkis = t.cast(gws.ext.helper.alkis.Object, self.root.find_first('gws.ext.helper.alkis'))

        if not self.alkis:
            gws.log.warn('alkissearch cannot init, alkis helper not found')
            return
        if not self.alkis.has_index:
            gws.log.warn('alkissearch cannot init, no alkis index')
            return

        self.valid = True

        self.limit: int = int(self.var('limit'))

        self.feature_templates: t.List[gws.ITemplate] = gws.base.template.bundle(self, self.var('templates'), _DEFAULT_FEATURE_TEMPLATES)

        self.feature_short_templates: t.List[gws.ITemplate] = [
            tpl for tpl in self.feature_templates
            if 'description' not in tpl.subject]

        self.print_template = self.create_child(
            'gws.ext.template',
            self.var('printTemplate', default=_DEFAULT_PRINT_TEMPLATE))

        self.ui: UiConfig = self.var('ui')

        p = self.var('export')
        g = gws.ext.helper.alkis.util.export.DEFAULT_GROUPS
        if not p and self.ui.useExport:
            p = gws.Config(groups=g)
        elif p:
            p.groups = p.groups or g
        self.export: gws.ext.helper.alkis.util.export.Config = p

        self.buchung: BuchungConfig = self.var('buchung')

        self.eigentuemer: EigentuemerConfig = self.var('eigentuemer')
        self.control_mode: bool = False
        self.log_table: str = ''
        self.control_rules: t.List[str] = []

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

        with_eigentuemer = self._can_read_eigentuemer(user)
        with_buchung = self._can_read_buchung(user)

        eg = {}

        if self.export and self._can_use_export(user):
            for n, g in enumerate(self.export.groups):
                if g.get('eigentuemer') and not with_eigentuemer:
                    continue
                if g.get('buchung') and not with_buchung:
                    continue
                eg[n] = g.get('title')

        return {
            'type': self.type,
            'exportGroups': eg,
            'gemarkungen': self.alkis.gemarkung_list(),
            'limit': self.limit,
            'printTemplate': self.print_template.props,
            'ui': self.ui,
            'withBuchung': with_buchung,
            'withControl': with_eigentuemer and self.control_mode,
            'withEigentuemer': with_eigentuemer,
            'withFlurnummer': self.alkis.has_flurnummer,
        }

    def api_get_toponyms(self, req: gws.IWebRequest, p: GetToponymsParams) -> GetToponymsResponse:
        """Return all Toponyms (Gemeinde/Gemarkung/Strasse) in the area"""

        self._validate_request(req, p)
        res = self.alkis.find_strasse(gws.ext.helper.alkis.FindStrasseQuery(p))

        gemeinde = {}
        gemarkung = {}

        for s in res.strassen:
            if s.gemeindeUid not in gemeinde:
                gemeinde[s.gemeindeUid] = ToponymGemeinde(name=re.sub(r'^Stadt\s+', '', s.gemeinde), uid=s.gemeindeUid)
            if s.gemarkungUid not in gemarkung:
                gemarkung[s.gemarkungUid] = ToponymGemarkung(name=s.gemarkung, uid=s.gemarkungUid, gemeindeUid=s.gemeindeUid)

        by_name = lambda x: x.name

        return GetToponymsResponse(
            gemeinden=sorted(gemeinde.values(), key=by_name),
            gemarkungen=sorted(gemarkung.values(), key=by_name),
            strasseNames=[s.strasse for s in res.strassen],
            strasseGemarkungUids=[s.gemarkungUid for s in res.strassen],
        )

    def api_find_flurstueck(self, req: gws.IWebRequest, p: FindFlurstueckParams) -> FindFlurstueckResponse:
        """Perform a Flurstueck search"""

        self._validate_request(req, p)
        return self._fetch_and_format(req, p, self.feature_short_templates, self.limit, self.limit)

    def api_get_details(self, req: gws.IWebRequest, p: GetDetailsParams) -> GetDetailsResponse:
        """Return a Flurstueck feature with details"""

        self._validate_request(req, p)
        res = self._fetch_and_format(req, p, self.feature_templates, 1, self.limit)

        if not res.features:
            raise gws.base.web.error.NotFound()

        return GetDetailsResponse(feature=res.features[0])

    def api_export(self, req: gws.IWebRequest, p: ExportParams) -> ExportResponse:
        """Export Flurstueck features"""

        self._validate_request(req, p)

        fp = p.findParams
        fp.projectUid = p.projectUid
        fp.locale = p.locale

        res = self._fetch(req, fp, self.limit)

        if not res.features:
            raise gws.base.web.error.NotFound()

        combined_rules = []

        for g in sorted(int(g) for g in p.groups):
            combined_rules.extend(self.export.groups[g].dataModel.rules)

        combined_model = self.root.create_unbound_object('gws.base.model', gws.Config(
            rules=combined_rules
        ))

        csv_bytes = gws.ext.helper.alkis.export.as_csv(self, res.features, combined_model)

        return ExportResponse(content=csv_bytes, mime='text/csv')

    def api_print(self, req: gws.IWebRequest, p: PrintParams) -> gws.base.print.job.StatusResponse:
        """Print Flurstueck features"""

        self._validate_request(req, p)

        fp = p.findParams
        fp.projectUid = p.projectUid
        fp.locale = p.locale

        res = self._fetch(req, fp, self.limit)

        if not res.features:
            raise gws.base.web.error.NotFound()

        pp = p.printParams
        pp.projectUid = p.projectUid
        pp.locale = p.locale
        pp.templateUid = self.print_template.uid
        pp.sections = []

        for feature in res.features:
            center = feature.shape.centroid
            pp.sections.append(gws.base.print.types.PrintSection(
                center=[center.x, center.y],
                context=feature.template_context,
                items=[
                    gws.base.print.types.PrintItemFeatures(
                        type='features',
                        features=[feature.props],
                        style=p.highlightStyle,
                    )
                ]
            ))

        job = gws.base.print.job.start(req, pp)
        return gws.base.print.job.status(job)

    ##

    def _validate_request(self, req: gws.IWebRequest, p: gws.Params):
        if not self.valid:
            raise gws.base.web.error.NotFound()
        req.require_project(p.projectUid)

    def _fetch_and_format(self, req, p: FindFlurstueckParams, templates: t.List[gws.ITemplate], soft_limit, hard_limit) -> FindFlurstueckResponse:
        fprops = []
        res = self._fetch(req, p, soft_limit, hard_limit)

        for f in res.features:
            f.apply_templates(templates)
            props = f.props
            del props.attributes
            fprops.append(props)

        return FindFlurstueckResponse(
            total=res.total,
            features=sorted(fprops, key=lambda f: f.elements['title']))

    def _fetch(self, req, p: FindFlurstueckParams, soft_limit=0, hard_limit=0) -> gws.ext.helper.alkis.FindFlurstueckResult:

        fq = gws.ext.helper.alkis.FindFlurstueckQuery(p)

        eigentuemer_flag = self._eigentuemer_flag(req, p)
        if eigentuemer_flag == _EF_FAIL:
            self._log_eigentuemer_access(req, p, is_ok=False)
            raise gws.base.web.error.BadRequest()

        fq.withEigentuemer = eigentuemer_flag == _EF_ALLOW
        fq.withBuchung = self._can_read_buchung(req.user)

        if p.get('shapes'):
            shape = gws.lib.shape.union(gws.lib.shape.from_props(s) for s in p.get('shapes'))
            if shape:
                fq.shape = shape.transformed_to(self.alkis.crs)

        if soft_limit:
            fq.limit = soft_limit

        fq.bblattMode = self.ui.get('bblattSearchMode', 'any')

        res = self.alkis.find_flurstueck(fq)

        gws.log.debug(f'FS_SEARCH ef={eigentuemer_flag} query={p!r} total={res.total!r} len={len(res.features)}')

        if hard_limit and res.total > hard_limit:
            raise gws.base.web.error.Conflict()

        project = req.require_project(p.projectUid)
        crs = p.get('crs') or project.map.crs

        for f in res.features:
            f.transform_to(crs)
            f.attributes.append(gws.Attribute(name='is_guest_user', value=req.user.is_guest))

        if fq.withEigentuemer:
            self._log_eigentuemer_access(req, p, is_ok=True, total=res.total, features=res.features)

        return res

    def _eigentuemer_flag(self, req: gws.IWebRequest, p: FindFlurstueckParams):
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

    def _log_eigentuemer_access(self, req: gws.IWebRequest, p: FindFlurstueckParams, is_ok, total=0, features=None):
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
            'date_time': gws.lib.date.now_iso(),
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

    def _can_read_eigentuemer(self, user: gws.IUser):
        b = user.can_use(self.eigentuemer)
        gws.log.debug(f'_can_read_eigentuemer user={user.fid!r} res={b}')
        return b

    def _can_read_buchung(self, user: gws.IUser):
        b = user.can_use(self.buchung)
        gws.log.debug(f'_can_read_buchung user={user.fid!r} res={b}')
        return b

    def _can_use_export(self, user: gws.IUser):
        return user.can_use(self.export, parent=self)
