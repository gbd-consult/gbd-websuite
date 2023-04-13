"""Backend for the Flurstücksuche (cadaster parlcels search) form."""

import os
import re

import gws
import gws.base.action
import gws.base.model
import gws.base.printer
import gws.base.storage
import gws.base.template
import gws.base.web.error
import gws.lib.date
import gws.base.feature
import gws.base.shape
import gws.lib.style
import gws.types as t

from . import provider, util, core

gws.ext.new.action('alkissearch')


STORAGE_CATEGORY = 'Alkis'

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


class EigentuemerConfig(gws.ConfigWithAccess):
    """Access to the Eigentümer (owner) information"""

    controlMode: bool = False 
    """restricted mode enabled"""
    controlRules: t.Optional[list[str]] 
    """regular expression for the restricted input control"""
    logTable: str = '' 
    """data access protocol table name"""


class EigentuemerOptions(gws.Node):
    control_mode: bool
    control_rules: list[str]
    log_table: str

    def configure(self):
        self.control_mode = self.cfg('controlMode')
        self.control_rules = self.cfg('controlRules', default=[])
        self.log_table = self.cfg('logTable')


##

class BuchungConfig(gws.ConfigWithAccess):
    """Access to the Grundbuch (register) information"""
    pass


class BuchungOptions(gws.Node):
    pass


##

class ExportGroupConfig(gws.Config):
    """Export group configuration"""

    dataModel: t.Optional[gws.base.model.Config] 
    """data model for this group"""
    title: str 
    """title for this group"""
    withBuchung: bool = False 
    """include Grundbuch (register) data"""
    withEigentuemer: bool = False 
    """include Eigentuemer (owner) data"""


class ExportConfig(gws.Config):
    """CSV Export configuration"""

    groups: t.Optional[list[ExportGroupConfig]] 
    """export groups"""


class ExportGroup(gws.Node):
    """Export group"""

    data_model: gws.base.model.Object
    with_buchung: bool
    with_eigentuemer: bool

    def configure(self):
        self.data_model = self.root.create_required(gws.ext.object.model, self.cfg('dataModel'))
        self.with_buchung = self.cfg('withBuchung')
        self.with_eigentuemer = self.cfg('withEigentuemer')
        self.title = self.cfg('title')


class ExportGroupProps(gws.Props):
    index: int
    title: str


_DEFAULT_EXPORT_GROUPS = [
    gws.Config(
        title='Basisdaten',
        dataModel=gws.Config(rules=[
            gws.Config(source='gemeinde', title='Gemeinde'),
            gws.Config(source='gemarkung_id', title='Gemarkung ID'),
            gws.Config(source='gemarkung', title='Gemarkung'),
            gws.Config(source='flurnummer', title='Flurnummer', type=gws.AttributeType.int),
            gws.Config(source='zaehler', title='Zähler', type=gws.AttributeType.int),
            gws.Config(source='nenner', title='Nenner'),
            gws.Config(source='flurstuecksfolge', title='Folge'),
            gws.Config(source='amtlicheflaeche', title='Fläche', type=gws.AttributeType.float),
            gws.Config(source='x', title='X', type=gws.AttributeType.float),
            gws.Config(source='y', title='Y', type=gws.AttributeType.float),
        ])
    ),
    gws.Config(
        title='Lage',
        dataModel=gws.Config(rules=[
            gws.Config(source='lage_strasse', title='FS Strasse'),
            gws.Config(source='lage_hausnummer', title='FS Hnr'),
        ])
    ),
    gws.Config(
        title='Gebäude',
        dataModel=gws.Config(rules=[
            gws.Config(source='gebaeude_area', title='Gebäude Fläche', type=gws.AttributeType.float),
            gws.Config(source='gebaeude_gebaeudefunktion', title='Gebäude Funktion'),
        ])
    ),
    gws.Config(
        title='Buchungsblatt', withBuchung=True,
        dataModel=gws.Config(rules=[
            gws.Config(source='buchung_buchungsart', title='Buchungsart'),
            gws.Config(source='buchung_buchungsblatt_blattart', title='Blattart'),
            gws.Config(source='buchung_buchungsblatt_buchungsblattkennzeichen', title='Blattkennzeichen'),
            gws.Config(source='buchung_buchungsblatt_buchungsblattnummermitbuchstabenerweiterung', title='Blattnummer'),
            gws.Config(source='buchung_laufendenummer', title='Laufende Nummer'),
        ])
    ),
    gws.Config(
        title='Eigentümer', withEigentuemer=True, withBuchung=True,
        dataModel=gws.Config(rules=[
            gws.Config(source='buchung_eigentuemer_person_vorname', title='Vorname'),
            gws.Config(source='buchung_eigentuemer_person_nachnameoderfirma', title='Name'),
            gws.Config(source='buchung_eigentuemer_person_geburtsdatum', title='Geburtsdatum'),
            gws.Config(source='buchung_eigentuemer_person_anschrift_strasse', title='Strasse'),
            gws.Config(source='buchung_eigentuemer_person_anschrift_hausnummer', title='Hnr'),
            gws.Config(source='buchung_eigentuemer_person_anschrift_postleitzahlpostzustellung', title='PLZ'),
            gws.Config(source='buchung_eigentuemer_person_anschrift_ort_post', title='Ort'),
        ])
    ),
    gws.Config(
        title='Nutzung',
        dataModel=gws.Config(rules=[
            gws.Config(source='nutzung_a_area', title='Nutzung Fläche', type=gws.AttributeType.float),
            gws.Config(source='nutzung_type', title='Nutzung Typ'),
        ])
    ),
]


##

class Config(provider.Config):
    """Flurstücksuche (cadaster parlcels search) action"""

    eigentuemer: t.Optional[EigentuemerConfig] 
    """access to the Eigentümer (owner) information"""
    buchung: t.Optional[BuchungConfig] 
    """access to the Grundbuch (register) information"""
    limit: int = 100 
    """search results limit"""
    templates: t.Optional[list[gws.ext.config.template]] 
    """templates for Flurstueck details"""
    ui: t.Optional[core.UiOptions] 
    """ui options"""
    export: t.Optional[ExportConfig] 
    """csv export configuration"""


##

class Props(gws.base.action.Props):
    exportGroups: list[ExportGroupProps]
    gemarkungen: list[core.Gemarkung]
    limit: int
    printTemplate: gws.base.template.Props
    ui: core.UiOptions
    withBuchung: bool
    withControl: bool
    withEigentuemer: bool
    withFlurnummer: bool


##

class BaseFindParams(gws.Request):
    gemarkung: str = ''
    gemarkungUid: str = ''
    gemeinde: str = ''
    gemeindeUid: str = ''
    strasse: str = ''
    strasseMode: t.Optional[core.StrasseQueryMode]


class GetToponymsParams(gws.Request):
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
    gemeinden: list[ToponymGemeinde]
    gemarkungen: list[ToponymGemarkung]
    strasseNames: list[str]
    strasseGemarkungUids: list[str]


class FindFlurstueckParams(BaseFindParams):
    wantEigentuemer: t.Optional[bool]
    controlInput: t.Optional[str]
    crs: t.Optional[gws.CrsName]
    shapes: t.Optional[list[gws.base.shape.Props]]

    bblatt: str = ''
    flaecheBis: str = ''
    flaecheVon: str = ''
    flurnummer: str = ''
    flurstuecksfolge: str = ''
    fsUids: list[str] = []
    hausnummer: str = ''
    name: str = ''
    nenner: str = ''
    vnum: str = ''
    vorname: str = ''
    zaehler: str = ''


class FindFlurstueckResponse(gws.Response):
    features: list[gws.FeatureProps]
    total: int


class FindAdresseParams(BaseFindParams):
    crs: t.Optional[gws.CrsName]

    bisHausnummer: str = ''
    hausnummer: str = ''
    kreis: str = ''
    kreisUid: str = ''
    land: str = ''
    landUid: str = ''
    regierungsbezirk: str = ''
    regierungsbezirkUid: str = ''


class FindAdresseResponse(gws.Response):
    features: list[gws.FeatureProps]
    total: int


class GetDetailsParams(FindFlurstueckParams):
    pass


class GetDetailsResponse(gws.Response):
    feature: gws.FeatureProps


class PrintParams(gws.Request):
    findParams: FindFlurstueckParams
    printParams: gws.base.printer.Request
    highlightStyle: gws.lib.style.Props


class ExportParams(gws.Request):
    findParams: FindFlurstueckParams
    groupIndexes: list[int]


class ExportResponse(gws.Response):
    content: str
    mime: str


##


_dir = os.path.dirname(__file__)

_DEFAULT_TEMPLATES = [
    gws.Config(
        subject='feature.title',
        type='html',
        text='{vollnummer}',
        access='all:allow',
    ),
    gws.Config(
        subject='feature.teaser',
        type='html',
        text='Flurstück {vollnummer}',
        access='all:allow',
    ),
    gws.Config(
        subject='feature.description',
        type='html',
        path=f'{_dir}/templates/data.cx.html',
        access='all:allow',
    ),
    gws.Config(
        subject='print',
        type='html',
        path=f'{_dir}/templates/print.cx.html',
        pageWidth=210,
        pageHeight=297,
        mapWidth=100,
        mapHeight=100,
        qualityLevels=[gws.TemplateQualityLevel(name='default', dpi=150)],
        access='all:allow',
    )
]

_EF_DENY = 0  # no access to Eigentümer
_EF_ALLOW = 1  # granted access to Eigentümer
_EF_FAIL = -1  # access to Eigentümer granted, control check failed


##


class Object(gws.base.action.Object):
    buchung: BuchungOptions
    eigentuemer: EigentuemerOptions
    export_groups: list[ExportGroup]
    limit: int
    print_template: gws.ITemplate
    provider: provider.Object
    templates: gws.ITemplateCollection
    ui: core.UiOptions

    def configure(self):
        self.provider = provider.create(self.root, self.config, shared=True)

        self.limit = self.cfg('limit')

        self.templates = gws.base.template.manager.create(
            self.root,
            items=self.cfg('templates'),
            defaults=_DEFAULT_TEMPLATES,
            parent=self)

        self.print_template = self.templates.find(subject='print')
        self.ui = self.cfg('ui')

        p = self.cfg('export')
        if p:
            groups = p.groups or _DEFAULT_EXPORT_GROUPS
        elif self.ui.useExport:
            groups = _DEFAULT_EXPORT_GROUPS
        else:
            groups = []
        self.export_groups = [self.root.create_required(ExportGroup, g) for g in groups]

        self.buchung = self.create_child(BuchungOptions, self.cfg('buchung'))
        self.eigentuemer = self.create_child(EigentuemerOptions, self.cfg('eigentuemer'))

        if self.eigentuemer.log_table:
            with self.provider.connection() as conn:
                if not conn.user_can('INSERT', self.eigentuemer.log_table):
                    raise gws.Error(f'no INSERT acccess to {self.eigentuemer.log_table!r}')

    def props(self, user):
        if not self.provider.has_index:
            return None

        with_eigentuemer = self._can_read_eigentuemer(user)
        with_buchung = self._can_read_buchung(user)

        export_groups = []

        for i, group in enumerate(self.export_groups):
            if group.with_eigentuemer and not with_eigentuemer:
                continue
            if group.with_buchung and not with_buchung:
                continue
            export_groups.append(ExportGroupProps(
                index=i, title=group.title
            ))

        return gws.merge(
            super().props(user),
            exportGroups=sorted(export_groups, key=lambda g: g.title),
            gemarkungen=self.provider.gemarkung_list(),
            limit=self.limit,
            printTemplate=self.print_template,
            ui=self.ui,
            withBuchung=with_buchung,
            withControl=with_eigentuemer and self.control_mode,
            withEigentuemer=with_eigentuemer,
            withFlurnummer=self.provider.has_flurnummer,
        )

    @gws.ext.command.api('alkissearchGetToponyms')
    def get_toponyms(self, req: gws.IWebRequester, p: GetToponymsParams) -> GetToponymsResponse:
        """Return all Toponyms (Gemeinde/Gemarkung/Strasse) in the area"""

        req.require_project(p.projectUid)
        res = self.provider.find_strasse(core.FindStrasseQuery(p))

        gemeinde = {}
        gemarkung = {}

        for s in res.strassen:
            if s.gemeindeUid not in gemeinde:
                gemeinde[s.gemeindeUid] = ToponymGemeinde(name=re.sub(r'^Stadt\s+', '', s.gemeinde), uid=s.gemeindeUid)
            if s.gemarkungUid not in gemarkung:
                gemarkung[s.gemarkungUid] = ToponymGemarkung(name=s.gemarkung, uid=s.gemarkungUid, gemeindeUid=s.gemeindeUid)

        return GetToponymsResponse(
            gemeinden=sorted(gemeinde.values(), key=lambda x: x.name),
            gemarkungen=sorted(gemarkung.values(), key=lambda x: x.name),
            strasseNames=[s.strasse for s in res.strassen],
            strasseGemarkungUids=[s.gemarkungUid for s in res.strassen],
        )

    @gws.ext.command.api('alkissearchFindFlurstueck')
    def find_flurstueck(self, req: gws.IWebRequester, p: FindFlurstueckParams) -> FindFlurstueckResponse:
        """Perform a Flurstueck search"""

        return self._find_and_format(req, p, ['feature.title', 'feature.teaser'], self.limit, self.limit)

    @gws.ext.command.api('alkissearchGetDetails')
    def get_details(self, req: gws.IWebRequester, p: GetDetailsParams) -> GetDetailsResponse:
        """Return a Flurstueck feature with details"""

        res = self._find_and_format(req, p, ['feature.title', 'feature.description'], 1, self.limit)

        if not res.features:
            raise gws.base.web.error.NotFound()

        return GetDetailsResponse(feature=res.features[0])

    @gws.ext.command.api('alkissearchExport')
    def export(self, req: gws.IWebRequester, p: ExportParams) -> ExportResponse:
        """Export Flurstueck features"""

        project = req.require_project(p.projectUid)

        fp = p.findParams
        fp.projectUid = project.uid
        fp.localeUid = p.localeUid

        res = self._find(req, fp, self.limit)

        if not res.features:
            raise gws.base.web.error.NotFound()

        combined_rules: list[gws.base.model.Rule] = []
        group_indexes = sorted(int(i) for i in p.groupIndexes)

        for i in group_indexes:
            combined_rules.extend(self.export_groups[i].data_model.rules)

        combined_model = self.create_child(
            gws.base.model.Object,
            gws.Config(rules=combined_rules),
            shared=True,
            key=group_indexes)

        csv_bytes = util.export.to_csv(self, res.features, combined_model)

        return ExportResponse(content=csv_bytes, mime='text/csv')

    @gws.ext.command.api('alkissearchPrint')
    def print(self, req: gws.IWebRequester, p: PrintParams) -> gws.base.printer.StatusResponse:
        """Print Flurstueck features"""

        project = req.require_project(p.projectUid)

        fp = p.findParams
        fp.projectUid = project.uid
        fp.localeUid = p.localeUid

        res = self._find(req, fp, self.limit)

        if not res.features:
            raise gws.base.web.error.NotFound()

        pp = p.printParams
        pp.projectUid = p.projectUid
        pp.localeUid = p.localeUid
        pp.templateUid = self.print_template.uid
        pp.sections = []

        for feature in res.features:
            if not feature.shape:
                gws.log.warning(f'feature {feature.uid!r} has no shape')
                continue
            center = feature.shape.centroid
            pp.sections.append(gws.base.printer.core.Section(
                center=[center.x, center.y],
                context=feature.template_context,
                items=[
                    gws.base.printer.core.ItemFeatures(
                        type='features',
                        features=[feature],
                        style=p.highlightStyle,
                    )
                ]
            ))

        job = gws.base.printer.job.start(req, pp)
        return gws.base.printer.job.status(job)

    @gws.ext.command.api('alkissearchStorage')
    def storage(self, req: gws.IWebRequester, p: gws.base.storage.Params) -> gws.base.storage.Response:
        helper: gws.base.storage.Object = self.root.app.require_helper('storage')
        return helper.handle_action(req, p, STORAGE_CATEGORY)

    ##

    def _ensure_index(self):
        if not self.provider.has_index:
            raise gws.Error('alkissearch cannot run, no alkis index')

    def _find_and_format(self, req, p: FindFlurstueckParams, template_subjects: list[str], soft_limit, hard_limit) -> FindFlurstueckResponse:
        feature_props_list = []
        res = self._find(req, p, soft_limit, hard_limit)

        for feature in res.features:
            feature.apply_templates(self.templates, subjects=template_subjects)
            f = gws.props(feature, req.user, self)
            if f:
                gws.pop(f, 'attributes')
                feature_props_list.append(f)

        return FindFlurstueckResponse(
            total=res.total,
            features=sorted(feature_props_list, key=lambda f: f.elements['title']))

    def _find(self, req, p: FindFlurstueckParams, soft_limit=0, hard_limit=0) -> core.FindFlurstueckResult:

        fq = core.FindFlurstueckQuery(p)

        eigentuemer_flag = self._eigentuemer_flag(req, p)
        if eigentuemer_flag == _EF_FAIL:
            self._log_eigentuemer_access(req, p, is_ok=False)
            raise gws.base.web.error.BadRequest()

        fq.withEigentuemer = eigentuemer_flag == _EF_ALLOW
        fq.withBuchung = self._can_read_buchung(req.user)

        if p.get('shapes'):
            shape = gws.base.shape.union([gws.base.shape.from_props(s) for s in p.get('shapes')])
            if shape:
                fq.shape = shape.transformed_to(self.provider.crs)

        if soft_limit:
            fq.limit = soft_limit

        fq.bblattMode = self.ui.get('bblattSearchMode', 'any')

        res = self.provider.find_flurstueck(fq)

        gws.log.debug(f'FS_SEARCH ef={eigentuemer_flag} query={p!r} total={res.total!r} len={len(res.features)}')

        if hard_limit and res.total > hard_limit:
            raise gws.base.web.error.Conflict()

        project = req.require_project(p.projectUid)
        crs = p.get('crs') or project.map.crs

        for f in res.features:
            f.transform_to(crs)
            f.attributes.append(gws.Attribute(name='is_guest_user', value=req.user.isGuest))

        if fq.withEigentuemer:
            self._log_eigentuemer_access(req, p, is_ok=True, total=res.total, features=res.features)

        return res

    def _eigentuemer_flag(self, req: gws.IWebRequester, p: FindFlurstueckParams):
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

    def _log_eigentuemer_access(self, req: gws.IWebRequester, p: FindFlurstueckParams, is_ok, total=0, features=None):
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
            'user_name': req.user.displayName,
            'control_input': (p.controlInput or '').strip(),
            'control_result': 1 if is_ok else 0,
            'fs_count': total,
            'fs_ids': fs_ids
        }

        with self.provider.connection() as conn:
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
        gws.log.debug(f'_can_read_eigentuemer res={b}')
        return b

    def _can_read_buchung(self, user: gws.IUser):
        b = user.can_use(self.buchung)
        gws.log.debug(f'_can_read_buchung res={b}')
        return b
