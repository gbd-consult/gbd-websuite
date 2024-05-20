"""Backend for the Flurstückssuche (cadaster parlcels search) form."""

from typing import Optional

import os
import re

import gws
import gws.base.action
import gws.base.database
import gws.base.feature
import gws.base.model
import gws.base.printer
import gws.base.shape
import gws.base.storage
import gws.base.template
import gws.base.web
import gws.config.util
import gws.lib.date
import gws.lib.sa as sa
import gws.lib.style


from .data import index, export
from .data import types as dt

gws.ext.new.action('alkis')


class EigentuemerConfig(gws.ConfigWithAccess):
    """Access to the Eigentümer (owner) information"""

    controlMode: bool = False
    """restricted mode enabled"""
    controlRules: Optional[list[str]]
    """regular expression for the restricted input control"""
    logTable: str = ''
    """data access protocol table name"""


class EigentuemerOptions(gws.Node):
    controlMode: bool
    controlRules: list[str]
    logTableName: str
    logTable: Optional[sa.Table]

    def configure(self):
        self.controlMode = self.cfg('controlMode')
        self.controlRules = self.cfg('controlRules', default=[])
        self.logTableName = self.cfg('logTable')
        self.logTable = None


class BuchungConfig(gws.ConfigWithAccess):
    """Access to the Grundbuch (register) information"""
    pass


class BuchungOptions(gws.Node):
    pass


class GemarkungListMode(gws.Enum):
    none = 'none'
    """do not show the list"""
    plain = 'plain'
    """only "gemarkung"""
    combined = 'combined'
    """"gemarkung (gemeinde)"""
    tree = 'tree'
    """a tree with level 1 = gemeinde and level 2 = gemarkung """


class StrasseListMode(gws.Enum):
    plain = 'plain'
    """just strasse"""
    withGemeinde = 'withGemeinde'
    """strasse (gemeinde)"""
    withGemarkung = 'withGemarkung'
    """strasse (gemarkung)"""
    withGemeindeIfRepeated = 'withGemeindeIfRepeated'
    """strasse (gemeinde), when needed for disambiguation """
    withGemarkungIfRepeated = 'withGemarkungIfRepeated'
    """strasse (gemarkung), when needed for disambiguation """


class Ui(gws.Config):
    """Flurstückssuche UI configuration."""

    useExport: bool = False
    """export function enabled"""
    useSelect: bool = False
    """select mode enabled"""
    usePick: bool = False
    """pick mode enabled"""
    useHistory: bool = False
    """history controls enabled"""
    searchSelection: bool = False
    """search in selection enabled"""
    searchSpatial: bool = False
    """spatial search enabled"""
    gemarkungListMode: GemarkungListMode = 'combined'
    """gemarkung list mode"""
    strasseListMode: StrasseListMode = 'plain'
    """strasse list entry format"""
    autoSpatialSearch: bool = False
    """activate spatial search after submit"""


class Config(gws.ConfigWithAccess):
    """Flurstückssuche action"""

    dbUid: str = ''
    """database provider ID"""
    crs: gws.CrsName
    """CRS for the ALKIS data"""
    dataSchema: str = 'public'
    """schema where ALKIS tables are stored"""
    indexSchema: str = 'gws8'
    """schema to store GWS internal indexes"""
    excludeGemarkung: Optional[list[str]]
    """Gemarkung (Administrative Unit) IDs to exclude from search"""

    eigentuemer: Optional[EigentuemerConfig] = {}
    """access to the Eigentümer (owner) information"""
    buchung: Optional[BuchungConfig] = {}
    """access to the Grundbuch (register) information"""
    limit: int = 100
    """search results limit"""
    templates: Optional[list[gws.ext.config.template]]
    """templates for Flurstueck details"""
    printers: Optional[list[gws.base.printer.Config]]
    """print configurations"""
    ui: Optional[Ui] = {}
    """ui options"""

    strasseSearchOptions: Optional[gws.TextSearchOptions]
    nameSearchOptions: Optional[gws.TextSearchOptions]
    buchungsblattSearchOptions: Optional[gws.TextSearchOptions]

    storage: Optional[gws.base.storage.Config]
    """storage configuration"""

    export: Optional[export.Config]
    """csv export configuration"""


##

class ExportGroupProps(gws.Props):
    index: int
    title: str


class Props(gws.base.action.Props):
    exportGroups: list[ExportGroupProps]
    limit: int
    printer: Optional[gws.base.printer.Props]
    ui: Ui
    storage: Optional[gws.base.storage.Props]
    withBuchung: bool
    withEigentuemer: bool
    withEigentuemerControl: bool
    withFlurnummer: bool


##

class GetToponymsRequest(gws.Request):
    pass


class GetToponymsResponse(gws.Response):
    gemeinde: list[list[str]]
    gemarkung: list[list[str]]
    strasse: list[list[str]]


class FindFlurstueckRequest(gws.Request):
    flurnummer: Optional[str]
    flurstuecksfolge: Optional[str]
    zaehler: Optional[str]
    nenner: Optional[str]
    fsnummer: Optional[str]

    flaecheBis: Optional[float]
    flaecheVon: Optional[float]

    gemarkung: Optional[str]
    gemarkungCode: Optional[str]
    gemeinde: Optional[str]
    gemeindeCode: Optional[str]
    kreis: Optional[str]
    kreisCode: Optional[str]
    land: Optional[str]
    landCode: Optional[str]
    regierungsbezirk: Optional[str]
    regierungsbezirkCode: Optional[str]

    strasse: Optional[str]
    hausnummer: Optional[str]

    bblatt: Optional[str]

    personName: Optional[str]
    personVorname: Optional[str]

    combinedFlurstueckCode: Optional[str]

    shapes: Optional[list[gws.base.shape.Props]]

    uids: Optional[list[str]]

    crs: Optional[gws.CrsName]
    eigentuemerControlInput: Optional[str]
    limit: Optional[int]

    wantEigentuemer: Optional[bool]
    wantHistorySearch: Optional[bool]
    wantHistoryDisplay: Optional[bool]

    displayThemes: Optional[list[dt.DisplayTheme]]


class FindFlurstueckResponse(gws.Response):
    features: list[gws.FeatureProps]
    total: int


class FindFlurstueckResult(gws.Data):
    flurstueckList: list[dt.Flurstueck]
    total: int
    query: dt.FlurstueckQuery


class FindAdresseRequest(gws.Request):
    crs: Optional[gws.Crs]

    gemarkung: Optional[str]
    gemarkungCode: Optional[str]
    gemeinde: Optional[str]
    gemeindeCode: Optional[str]
    kreis: Optional[str]
    kreisCode: Optional[str]
    land: Optional[str]
    landCode: Optional[str]
    regierungsbezirk: Optional[str]
    regierungsbezirkCode: Optional[str]

    strasse: Optional[str]
    hausnummer: Optional[str]
    bisHausnummer: Optional[str]
    hausnummerNotNull: Optional[bool]

    wantHistorySearch: Optional[bool]

    combinedAdresseCode: Optional[str]


class FindAdresseResponse(gws.Response):
    features: list[gws.FeatureProps]
    total: int


class PrintFlurstueckRequest(gws.Request):
    findRequest: FindFlurstueckRequest
    printRequest: gws.PrintRequest
    featureStyle: gws.StyleProps


class ExportFlurstueckRequest(gws.Request):
    findRequest: FindFlurstueckRequest
    groupIndexes: list[int]


class ExportFlurstueckResponse(gws.Response):
    content: str
    mime: str


##


_dir = os.path.dirname(__file__)

_DEFAULT_TEMPLATES = [
    gws.Config(
        subject='flurstueck.title',
        type='html',
        path=f'{_dir}/templates/title.cx.html',
    ),
    gws.Config(
        subject='flurstueck.teaser',
        type='html',
        path=f'{_dir}/templates/title.cx.html',
    ),
    gws.Config(
        subject='flurstueck.label',
        type='html',
        path=f'{_dir}/templates/title.cx.html',
    ),
    gws.Config(
        subject='flurstueck.description',
        type='html',
        path=f'{_dir}/templates/description.cx.html',
    ),
    gws.Config(
        subject='adresse.title',
        type='html',
        path=f'{_dir}/templates/adresse_title.cx.html',
    ),
    gws.Config(
        subject='adresse.teaser',
        type='html',
        path=f'{_dir}/templates/adresse_title.cx.html',
    ),
    gws.Config(
        subject='adresse.label',
        type='html',
        path=f'{_dir}/templates/adresse_title.cx.html',
    ),
]

_DEFAULT_PRINTER = gws.Config(
    uid='gws.plugin.alkis.default_printer',
    template=gws.Config(
        type='html',
        path=f'{_dir}/templates/print.cx.html',
    ),
    qualityLevels=[{'dpi': 72}],
)


##

class Model(gws.base.model.dynamic_model.Object):
    def configure(self):
        self.uidName = 'uid'
        self.geometryName = 'geometry'
        self.loadingStrategy = gws.FeatureLoadingStrategy.all


class Object(gws.base.action.Object):
    dbProvider: gws.DatabaseProvider

    ix: index.Object
    ixStatus: index.Status

    buchung: BuchungOptions
    eigentuemer: EigentuemerOptions

    dataSchema: str
    excludeGemarkung: set[str]

    model: gws.Model
    ui: Ui
    limit: int

    templates: list[gws.Template]
    printers: list[gws.Printer]

    export: Optional[export.Object]

    strasseSearchOptions: gws.TextSearchOptions
    nameSearchOptions: gws.TextSearchOptions
    buchungsblattSearchOptions: gws.TextSearchOptions

    storage: Optional[gws.base.storage.Object]

    def configure(self):
        gws.config.util.configure_database_provider_for(self, ext_type='postgres', required=True)
        self.ix = self.root.create_shared(
            index.Object,
            _defaultProvider=self.dbProvider,
            crs=self.cfg('crs'),
            schema=self.cfg('indexSchema'),
            excludeGemarkung=self.cfg('excludeGemarkung'),
            uid='gws.plugin.alkis.data.index.' + self.cfg('indexSchema')
        )

        self.limit = self.cfg('limit')
        self.model = self.create_child(Model)
        self.ui = self.cfg('ui')

        self.dataSchema = self.cfg('dataSchema')
        self.excludeGemarkung = set(self.cfg('excludeGemarkung', default=[]))

        p = self.cfg('templates', default=[]) + _DEFAULT_TEMPLATES
        self.templates = [self.create_child(gws.ext.object.template, c) for c in p]

        self.printers = self.create_children(gws.ext.object.printer, self.cfg('printers'))
        self.printers.append(self.root.create_shared(gws.ext.object.printer, _DEFAULT_PRINTER))

        d = gws.TextSearchOptions(type='exact')
        self.strasseSearchOptions = self.cfg('strasseSearchOptions', default=d)
        self.nameSearchOptions = self.cfg('nameSearchOptions', default=d)
        self.buchungsblattSearchOptions = self.cfg('buchungsblattSearchOptions', default=d)

        self.buchung = self.create_child(BuchungOptions, self.cfg('buchung'))

        self.eigentuemer = self.create_child(EigentuemerOptions, self.cfg('eigentuemer'))
        if self.eigentuemer.logTableName:
            self.eigentuemer.logTable = self.ix.provider.table(self.eigentuemer.logTableName)

        self.storage = self.create_child_if_configured(
            gws.base.storage.Object, self.cfg('storage'), categoryName='Alkis')

        p = self.cfg('export')
        if p:
            self.export = self.create_child(export.Object, p)
        elif self.ui.useExport:
            self.export = self.create_child(export.Object)
        else:
            self.export = None

    def activate(self):
        self.ixStatus = self.ix.status()

    def props(self, user):
        if not self.ixStatus.basic:
            return None

        export_groups = []
        if self.ui.useExport:
            export_groups = [ExportGroupProps(title=g.title, index=g.index) for g in self._export_groups(user)]

        return gws.u.merge(
            super().props(user),
            exportGroups=export_groups,
            limit=self.limit,
            printer=gws.u.first(p for p in self.printers if user.can_use(p)),
            ui=self.ui,
            storage=self.storage,
            withBuchung=(
                    self.ixStatus.buchung
                    and user.can_read(self.buchung)
            ),
            withEigentuemer=(
                    self.ixStatus.eigentuemer
                    and user.can_read(self.eigentuemer)
            ),
            withEigentuemerControl=(
                    self.ixStatus.eigentuemer
                    and user.can_read(self.eigentuemer)
                    and self.eigentuemer.controlMode
            ),
        )

    @gws.ext.command.api('alkisGetToponyms')
    def get_toponyms(self, req: gws.WebRequester, p: GetToponymsRequest) -> GetToponymsResponse:
        """Return all Toponyms (Gemeinde/Gemarkung/Strasse) in the area"""

        req.user.require_project(p.projectUid)

        gemeinde_dct = {}
        gemarkung_dct = {}
        strasse_lst = []

        for s in self.ix.strasse_list():
            gemeinde_dct[s.gemeinde.code] = [s.gemeinde.text, s.gemeinde.code]
            gemarkung_dct[s.gemarkung.code] = [s.gemarkung.text, s.gemarkung.code, s.gemeinde.code]
            strasse_lst.append([s.name, s.gemeinde.code, s.gemarkung.code])

        return GetToponymsResponse(
            gemeinde=sorted(gemeinde_dct.values()),
            gemarkung=sorted(gemarkung_dct.values()),
            strasse=sorted(strasse_lst)
        )

    @gws.ext.command.api('alkisFindAdresse')
    def find_adresse(self, req: gws.WebRequester, p: FindAdresseRequest) -> FindAdresseResponse:
        """Perform an Adresse search."""

        project = req.user.require_project(p.projectUid)
        crs = p.get('crs') or project.map.bounds.crs

        ad_list, query = self.find_adresse_objects(req, p)
        if not ad_list:
            return FindAdresseResponse(
                features=[],
                total=0,
            )

        templates = [
            self.root.app.templateMgr.find_template(self, user=req.user, subject='adresse.title'),
            self.root.app.templateMgr.find_template(self, user=req.user, subject='adresse.teaser'),
            self.root.app.templateMgr.find_template(self, user=req.user, subject='adresse.label'),
        ]

        fprops = []
        mc = gws.ModelContext(op=gws.ModelOperation.read, readMode=gws.ModelReadMode.render, user=req.user)

        for ad in ad_list:
            f = gws.base.feature.new(model=self.model)
            f.attributes = vars(ad)
            f.attributes['geometry'] = f.attributes.pop('shape')
            f.transform_to(crs)
            f.render_views(templates, user=req.user)
            fprops.append(self.model.feature_to_view_props(f, mc))

        return FindAdresseResponse(
            features=fprops,
            total=len(ad_list),
        )

    @gws.ext.command.api('alkisFindFlurstueck')
    def find_flurstueck(self, req: gws.WebRequester, p: FindFlurstueckRequest) -> FindFlurstueckResponse:
        """Perform a Flurstueck search"""

        project = req.user.require_project(p.projectUid)
        crs = p.get('crs') or project.map.bounds.crs

        fs_list, query = self.find_flurstueck_objects(req, p)
        if not fs_list:
            return FindFlurstueckResponse(
                features=[],
                total=0,
            )

        templates = [
            self.root.app.templateMgr.find_template(self, user=req.user, subject='flurstueck.title'),
            self.root.app.templateMgr.find_template(self, user=req.user, subject='flurstueck.teaser'),
        ]

        if query.options.displayThemes:
            templates.append(
                self.root.app.templateMgr.find_template(self, req.user, subject='flurstueck.description'),
            )

        args = dict(
            withHistory=query.options.withHistoryDisplay,
            withDebug=bool(self.root.app.developer_option('alkis.debug_templates')),
        )

        fprops = []
        mc = gws.ModelContext(op=gws.ModelOperation.read, readMode=gws.ModelReadMode.render, user=req.user)

        for fs in fs_list:
            f = gws.base.feature.new(model=self.model)
            f.attributes = dict(uid=fs.uid, fs=fs, geometry=fs.shape)
            f.transform_to(crs)
            f.render_views(templates, user=req.user, **args)
            f.attributes.pop('fs')
            fprops.append(self.model.feature_to_view_props(f, mc))

        fprops.sort(key=lambda p: p.views['title'])

        return FindFlurstueckResponse(
            features=fprops,
            total=len(fs_list),
        )

    @gws.ext.command.api('alkisExportFlurstueck')
    def export_flurstueck(self, req: gws.WebRequester, p: ExportFlurstueckRequest) -> ExportFlurstueckResponse:
        if not self.export:
            raise gws.NotFoundError()

        groups = [g for g in self._export_groups(req.user) if g.index in p.groupIndexes]
        if not groups:
            raise gws.NotFoundError()

        find_request = p.findRequest
        find_request.projectUid = p.projectUid

        fs_list, _ = self.find_flurstueck_objects(req, find_request)
        if not fs_list:
            raise gws.NotFoundError()

        csv_bytes = self.export.export_as_csv(fs_list, groups, req.user)

        return ExportFlurstueckResponse(content=csv_bytes, mime='text/csv')

    @gws.ext.command.api('alkisPrintFlurstueck')
    def print_flurstueck(self, req: gws.WebRequester, p: PrintFlurstueckRequest) -> gws.PrintJobResponse:
        """Print Flurstueck features"""

        project = req.user.require_project(p.projectUid)

        find_request = p.findRequest
        find_request.projectUid = p.projectUid

        fs_list, query = self.find_flurstueck_objects(req, find_request)
        if not fs_list:
            raise gws.NotFoundError()

        print_request = p.printRequest
        print_request.projectUid = p.projectUid
        crs = print_request.get('crs') or project.map.bounds.crs

        templates = [
            self.root.app.templateMgr.find_template(self, user=req.user, subject='flurstueck.label'),
        ]

        base_map = print_request.maps[0]
        fs_maps = []

        mc = gws.ModelContext(op=gws.ModelOperation.read, readMode=gws.ModelReadMode.render, user=req.user)

        for fs in fs_list:
            f = gws.base.feature.new(model=self.model)
            f.attributes = dict(uid=fs.uid, fs=fs, geometry=fs.shape)
            f.transform_to(crs)
            f.render_views(templates, user=req.user)
            f.cssSelector = p.featureStyle.cssSelector

            c = f.shape().centroid()
            fs_map = gws.PrintMap(base_map)
            # @TODO scale to fit the fs?
            fs_map.center = (c.x, c.y)
            fs_plane = gws.PrintPlane(
                type=gws.PrintPlaneType.features,
                features=[self.model.feature_to_view_props(f, mc)],
            )
            fs_map.planes = [fs_plane] + base_map.planes
            fs_maps.append(fs_map)

        print_request.maps = fs_maps
        print_request.args = dict(
            flurstueckList=fs_list,
            withHistory=query.options.withHistoryDisplay,
            withDebug=bool(self.root.app.developer_option('alkis.debug_templates')),
        )

        job = self.root.app.printerMgr.start_job(print_request, req.user)
        return self.root.app.printerMgr.status(job)

    @gws.ext.command.api('alkisSelectionStorage')
    def handle_storage(self, req: gws.WebRequester, p: gws.base.storage.Request) -> gws.base.storage.Response:
        if not self.storage:
            raise gws.base.web.error.NotFound()
        return self.storage.handle_request(req, p)

    ##

    def find_flurstueck_objects(self, req: gws.WebRequester, p: FindFlurstueckRequest) -> tuple[list[dt.Flurstueck], dt.FlurstueckQuery]:
        query = self._prepare_flurstueck_query(req, p)
        fs_list = self.ix.find_flurstueck(query)

        if query.options.withEigentuemer:
            self._log_eigentuemer_access(
                req,
                p.eigentuemerControlInput,
                is_ok=True,
                total=len(fs_list),
                fs_uids=[fs.uid for fs in fs_list]
            )

        return fs_list, query

    def find_adresse_objects(self, req: gws.WebRequester, p: FindAdresseRequest) -> tuple[list[dt.Adresse], dt.AdresseQuery]:
        query = self._prepare_adresse_query(req, p)
        ad_list = self.ix.find_adresse(query)
        return ad_list, query

    FLURSTUECK_QUERY_FIELDS = [
        'flurnummer',
        'flurstuecksfolge',
        'zaehler',
        'nenner',
        'flurstueckskennzeichen',
        'flaecheBis',
        'flaecheVon',
        'gemarkung',
        'gemarkungCode',
        'gemeinde',
        'gemeindeCode',
        'kreis',
        'kreisCode',
        'land',
        'landCode',
        'regierungsbezirk',
        'regierungsbezirkCode',
        'strasse',
        'hausnummer',
        'personName',
        'personVorname',
        'uids',
    ]
    ADRESSE_QUERY_FIELDS = [
        'gemarkung',
        'gemarkungCode',
        'gemeinde',
        'gemeindeCode',
        'kreis',
        'kreisCode',
        'land',
        'landCode',
        'regierungsbezirk',
        'regierungsbezirkCode',
        'strasse',
        'hausnummer',
        'bisHausnummer',
        'hausnummerNotNull',
    ]
    COMBINED_FLURSTUECK_FIELDS = [
        'landCode', 'gemarkungCode', 'flurnummer', 'zaehler', 'nenner', 'flurstuecksfolge'
    ]
    COMBINED_ADRESSE_FIELDS = [
        'strasse', 'hausnummer', 'plz', 'gemeinde', 'bisHausnummer'
    ]

    def _prepare_flurstueck_query(self, req: gws.WebRequester, p: FindFlurstueckRequest) -> dt.FlurstueckQuery:
        query = dt.FlurstueckQuery()

        for f in self.FLURSTUECK_QUERY_FIELDS:
            setattr(query, f, getattr(p, f, None))

        if p.combinedFlurstueckCode:
            self._query_combined_code(query, p.combinedFlurstueckCode, self.COMBINED_FLURSTUECK_FIELDS)

        if p.fsnummer:
            self._query_fsnummer(query, p.fsnummer)

        if p.shapes:
            shapes = [gws.base.shape.from_props(s) for s in p.shapes]
            query.shape = shapes[0] if len(shapes) == 1 else shapes[0].union(shapes[1:])

        if p.bblatt:
            query.buchungsblattkennzeichenList = p.bblatt.replace(';', ' ').replace(',', ' ').strip().split()

        options = dt.FlurstueckQueryOptions(
            strasseSearchOptions=self.strasseSearchOptions,
            nameSearchOptions=self.nameSearchOptions,
            buchungsblattSearchOptions=self.buchungsblattSearchOptions,
            hardLimit=self.limit,
            withEigentuemer=False,
            withBuchung=False,
            withHistorySearch=bool(p.wantHistorySearch),
            withHistoryDisplay=bool(p.wantHistoryDisplay),
            displayThemes=p.displayThemes or [],
        )

        want_eigentuemer = (
                p.wantEigentuemer
                or dt.DisplayTheme.eigentuemer in options.displayThemes
                or any(getattr(query, f) is not None for f in dt.EigentuemerAccessRequired)
        )
        if want_eigentuemer:
            self._check_eigentuemer_access(req, p.eigentuemerControlInput)
            options.withEigentuemer = True

        want_buchung = (
                dt.DisplayTheme.buchung in options.displayThemes
                or any(getattr(query, f) is not None for f in dt.BuchungAccessRequired)
        )
        if want_buchung:
            self._check_buchung_access(req, p.eigentuemerControlInput)
            options.withBuchung = True

        query.options = options
        return query

    def _prepare_adresse_query(self, req: gws.WebRequester, p: FindAdresseRequest) -> dt.AdresseQuery:
        query = dt.AdresseQuery()

        for f in self.ADRESSE_QUERY_FIELDS:
            setattr(query, f, getattr(p, f, None))

        if p.combinedAdresseCode:
            self._query_combined_code(query, p.combinedAdresseCode, self.COMBINED_ADRESSE_FIELDS)

        options = dt.AdresseQueryOptions(
            strasseSearchOptions=self.strasseSearchOptions,
            withHistorySearch=bool(p.wantHistorySearch),
        )

        query.options = options
        return query

    def _query_fsnummer(self, query: dt.FlurstueckQuery, vn: str):

        if vn.startswith('DE'):
            # search by gml_id
            query.uids = query.uids or []
            query.uids.append(vn)
            return

        if re.match('^[0-9_]+$', vn) and len(vn) >= 14:
            # search by fs kennzeichen
            # the length must be at least 2+4+3+5
            # (see gdi6, AX_Flurstueck_Kerndaten.flurstueckskennzeichen)
            query.flurstueckskennzeichen = vn
            return

        # search by a compound fs number
        parts = index.parse_fsnummer(vn)
        if parts is None:
            raise gws.BadRequestError(f'invalid fsnummer {vn!r}')
        query.update(parts)

    def _query_combined_code(self, query: dt.FlurstueckQuery | dt.AdresseQuery, code_value: str, code_fields: list[str]):
        for val, field in zip(code_value.split('_'), code_fields):
            if val and val != '0':
                setattr(query, field, val)

    ##

    def _export_groups(self, user):
        if not self.export:
            return []

        groups = []

        for g in self.export.groups:
            if g.withBuchung and not user.can_read(self.buchung):
                continue
            if g.withEigentuemer and not user.can_read(self.eigentuemer):
                continue
            groups.append(g)

        return groups

    def _check_eigentuemer_access(self, req: gws.WebRequester, control_input: str):
        if not req.user.can_read(self.eigentuemer):
            raise gws.ForbiddenError('cannot read eigentuemer')
        if self.eigentuemer.controlMode and not self._check_eigentuemer_control_input(control_input):
            self._log_eigentuemer_access(req, is_ok=False, control_input=control_input)
            raise gws.ForbiddenError('eigentuemer control input failed')

    def _check_buchung_access(self, req: gws.WebRequester, control_input: str):
        if not req.user.can_read(self.buchung):
            raise gws.ForbiddenError('cannot read buchung')

    def _log_eigentuemer_access(self, req: gws.WebRequester, control_input: str, is_ok: bool, total=None, fs_uids=None):
        if self.eigentuemer.logTable is None:
            return

        if self._eigentuemerLogTable is None:
            schema, name = self.ix.provider.split_table_name(self.eigentuemer.logTable)
            self._eigentuemerLogTable = sa.Table(
                name,
                self.ix.saMeta,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('app_name', sa.Text),
                sa.Column('date_time', sa.DateTime),
                sa.Column('ip', sa.Text),
                sa.Column('login', sa.Text),
                sa.Column('user_name', sa.Text),
                sa.Column('control_input', sa.Text),
                sa.Column('control_result', sa.Integer),
                sa.Column('fs_count', sa.Integer),
                sa.Column('fs_ids', sa.Text),
                schema=schema
            )

        data = dict(
            app_name='gws',
            date_time=gws.lib.date.now_iso(),
            ip=req.env('REMOTE_ADDR', ''),
            login=req.user.uid,
            user_name=req.user.displayName,
            control_input=(control_input or '').strip(),
            control_result=1 if is_ok else 0,
            fs_count=total or 0,
            fs_ids=','.join(fs_uids or []),
        )

        with self.ix.connect() as conn:
            conn.execute(sa.insert(self.eigentuemer.logTable).values([data]))
            conn.commit()

        gws.log.debug(f'_log_eigentuemer_access {is_ok=}')

    def _check_eigentuemer_control_input(self, control_input):
        if not self.eigentuemer.controlRules:
            return True

        control_input = (control_input or '').strip()

        for rule in self.eigentuemer.controlRules:
            if re.search(rule, control_input):
                return True

        return False
