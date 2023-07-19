"""Backend for the Flurstückssuche (cadaster parlcels search) form."""

import os
import re

import gws
import gws.base.action
import gws.base.database
import gws.base.feature
import gws.base.model
import gws.base.printer
import gws.base.printer.core
import gws.base.printer.job
import gws.base.shape
import gws.base.template
import gws.base.web
import gws.lib.date
import gws.lib.style
import gws.lib.sa as sa

import gws.types as t

from .data import index
from .data import types as dt

gws.ext.new.action('alkis')


class EigentuemerConfig(gws.ConfigWithAccess):
    """Access to the Eigentümer (owner) information"""

    controlMode: bool = False
    """restricted mode enabled"""
    controlRules: t.Optional[list[str]]
    """regular expression for the restricted input control"""
    logTable: str = ''
    """data access protocol table name"""


class EigentuemerOptions(gws.Node):
    controlMode: bool
    controlRules: list[str]
    logTable: str
    saTable: sa.Table

    def configure(self):
        self.controlMode = self.cfg('controlMode')
        self.controlRules = self.cfg('controlRules', default=[])
        self.logTable = self.cfg('logTable')


class BuchungConfig(gws.ConfigWithAccess):
    """Access to the Grundbuch (register) information"""
    pass


class BuchungOptions(gws.Node):
    pass


class GemarkungListMode(t.Enum):
    none = 'none'
    """do not show the list"""
    plain = 'plain'
    """only "gemarkung"""
    combined = 'combined'
    """"gemarkung (gemeinde)"""
    tree = 'tree'
    """a tree with level 1 = gemeinde and level 2 = gemarkung """


class StrasseListMode(t.Enum):
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
    excludeGemarkung: t.Optional[list[str]]
    """Gemarkung (Administrative Unit) IDs to exclude from search"""

    eigentuemer: t.Optional[EigentuemerConfig] = {}
    """access to the Eigentümer (owner) information"""
    buchung: t.Optional[BuchungConfig] = {}
    """access to the Grundbuch (register) information"""
    limit: int = 100
    """search results limit"""
    templates: t.Optional[list[gws.ext.config.template]]
    """templates for Flurstueck details"""
    ui: t.Optional[Ui] = {}
    """ui options"""
    strasseSearchOptions: t.Optional[gws.TextSearchOptions]
    nameSearchOptions: t.Optional[gws.TextSearchOptions]
    buchungsblattSearchOptions: t.Optional[gws.TextSearchOptions]

    # export: t.Optional[ExportConfig] 
    # """csv export configuration"""


class Props(gws.base.action.Props):
    # exportGroups: list[ExportGroupProps]
    limit: int
    printTemplate: gws.base.template.Props
    ui: Ui
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
    flurnummer: t.Optional[str]
    flurstuecksfolge: t.Optional[str]
    zaehler: t.Optional[str]
    nenner: t.Optional[str]
    vollNummer: t.Optional[str]

    flaecheBis: t.Optional[float]
    flaecheVon: t.Optional[float]

    gemarkungCode: t.Optional[str]
    gemeindeCode: t.Optional[str]

    strasse: t.Optional[str]
    hausnummer: t.Optional[str]

    bblatt: t.Optional[str]

    personName: t.Optional[str]
    personVorname: t.Optional[str]

    shapes: t.Optional[list[gws.base.shape.Props]]

    uids: t.Optional[list[str]]

    crs: t.Optional[gws.CrsName]
    eigentuemerControlInput: t.Optional[str]
    limit: t.Optional[int]

    wantHistorySearch: t.Optional[bool]
    wantHistoryDisplay: t.Optional[bool]

    displayThemes: t.Optional[list[dt.DisplayTheme]]


class FindFlurstueckResult(gws.Data):
    flurstueckList: list[dt.Flurstueck]
    total: int
    query: dt.FlurstueckSearchQuery
    queryOptions: dt.FlurstueckSearchOptions


class FindFlurstueckResponse(gws.Response):
    features: list[gws.FeatureProps]
    total: int


class FindAdresseRequest():
    crs: t.Optional[gws.ICrs]

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


class PrintFlurstueckRequest(gws.Request):
    findRequest: FindFlurstueckRequest
    printRequest: gws.base.printer.Request
    highlightStyle: gws.lib.style.Props


class ExportRequest(gws.Request):
    findRequest: FindFlurstueckRequest
    groups: list[str]


class ExportResponse(gws.Response):
    content: str
    mime: str


##


_dir = os.path.dirname(__file__)

_DEFAULT_TEMPLATES = [
    gws.Config(
        subject='feature.title',
        type='html',
        path=f'{_dir}/templates/title.cx.html',
    ),
    gws.Config(
        subject='feature.teaser',
        type='html',
        path=f'{_dir}/templates/title.cx.html',
    ),
    gws.Config(
        subject='feature.label',
        type='html',
        path=f'{_dir}/templates/label.cx.html',
    ),
    gws.Config(
        subject='feature.description',
        type='html',
        path=f'{_dir}/templates/description.cx.html',
    ),
    gws.Config(
        subject='feature.print',
        type='html',
        path=f'{_dir}/templates/print.cx.html',
        pageWidth=210,
        pageHeight=297,
        mapWidth=100,
        mapHeight=100,
        qualityLevels=[gws.TemplateQualityLevel(name='default', dpi=150)],
    ),
]


##

class Model(gws.base.model.Object):
    def configure(self):
        self.keyName = 'uid'
        self.geometryName = 'geometry'
        self.loadingStrategy = gws.FeatureLoadingStrategy.all


class Object(gws.base.action.Object):
    index: index.Object
    indexExists: bool
    dataSchema: str

    buchung: BuchungOptions
    eigentuemer: EigentuemerOptions

    # export_groups: list[ExportGroup]
    limit: int

    templates: list[gws.ITemplate]

    model: gws.IModel

    ui: Ui

    strasseSearchOptions: gws.TextSearchOptions
    nameSearchOptions: gws.TextSearchOptions
    buchungsblattSearchOptions: gws.TextSearchOptions

    def configure(self):
        provider = gws.base.database.provider.get_for(self, ext_type='postgres')

        self.index = self.root.create_shared(
            index.Object,
            provider=provider,
            crs=self.cfg('crs'),
            schema=self.cfg('indexSchema'),
            uid='gws.plugin.alkis.data.index.' + self.cfg('indexSchema')
        )

        self.indexExists = self.index.exists()
        if self.indexExists:
            gws.log.info(f'alkis index ok')
        else:
            gws.log.warning(f'alkis index NOT FOUND')

        self.limit = self.cfg('limit')
        self.model = self.create_child(Model)
        self.ui = self.cfg('ui')

        p = self.cfg('templates', default=[]) + _DEFAULT_TEMPLATES
        self.templates = [self.create_child(gws.ext.object.template, c) for c in p]

        d = gws.TextSearchOptions(type='exact')
        self.strasseSearchOptions = self.cfg('strasseSearchOptions', default=d)
        self.nameSearchOptions = self.cfg('nameSearchOptions', default=d)
        self.buchungsblattSearchOptions = self.cfg('buchungsblattSearchOptions', default=d)

        self.buchung = self.create_child(BuchungOptions, self.cfg('buchung'))
        self.eigentuemer = self.create_child(EigentuemerOptions, self.cfg('eigentuemer'))

        self.dataSchema = self.cfg('dataSchema')

        #
        # p = self.cfg('export')
        # if p:
        #     groups = p.groups or _DEFAULT_EXPORT_GROUPS
        # elif self.ui.useExport:
        #     groups = _DEFAULT_EXPORT_GROUPS
        # else:
        #     groups = []
        # self.export_groups = [self.root.create_required(ExportGroup, g) for g in groups]

    def props(self, user):
        if not self.indexExists:
            return None

        # export_groups = []
        #
        # for i, group in enumerate(self.export_groups):
        #     if group.with_eigentuemer and not with_eigentuemer:
        #         continue
        #     if group.with_buchung and not with_buchung:
        #         continue
        #     export_groups.append(ExportGroupProps(
        #         index=i, title=group.title
        #     ))

        return gws.merge(
            super().props(user),
            # exportGroups=sorted(export_groups, key=lambda g: g.title),
            limit=self.limit,
            printTemplate=gws.base.template.locate(self, user=user, subject='feature.print'),
            ui=self.ui,
            withBuchung=user.can_read(self.buchung),
            withEigentuemer=user.can_read(self.eigentuemer),
            withEigentuemerControl=user.can_read(self.eigentuemer) and self.eigentuemer.controlMode,
            # withFlurnummer=self.provider.has_flurnummer,
        )

    @gws.ext.command.api('alkisGetToponyms')
    def get_toponyms(self, req: gws.IWebRequester, p: GetToponymsRequest) -> GetToponymsResponse:
        """Return all Toponyms (Gemeinde/Gemarkung/Strasse) in the area"""

        req.require_project(p.projectUid)

        gemeinde_dct = {}
        gemarkung_dct = {}
        strasse_lst = []

        for s in self.index.strasse_list():
            gemeinde_dct[s.gemeinde.code] = [s.gemeinde.text, s.gemeinde.code]
            gemarkung_dct[s.gemarkung.code] = [s.gemarkung.text, s.gemarkung.code, s.gemeinde.code]
            strasse_lst.append([s.name, s.gemeinde.code, s.gemarkung.code])

        return GetToponymsResponse(
            gemeinde=sorted(gemeinde_dct.values()),
            gemarkung=sorted(gemarkung_dct.values()),
            strasse=sorted(strasse_lst)
        )

    @gws.ext.command.api('alkisFindFlurstueck')
    def find_flurstueck(self, req: gws.IWebRequester, p: FindFlurstueckRequest) -> FindFlurstueckResponse:
        """Perform a Flurstueck search"""

        project = req.require_project(p.projectUid)
        crs = p.get('crs') or project.map.bounds.crs

        res = self._find_flurstueck(req, p)
        if not res.flurstueckList:
            return FindFlurstueckResponse(
                features=[],
                total=0,
            )

        templates = [
            gws.base.template.locate(self, user=req.user, subject='feature.title'),
            gws.base.template.locate(self, user=req.user, subject='feature.teaser'),
        ]

        if res.queryOptions.displayThemes:
            templates.append(
                gws.base.template.locate(self, req.user, subject='feature.description'),
            )

        args = dict(
            withHistory=res.queryOptions.withHistoryDisplay,
            withDebug=bool(self.root.app.developer_option('alkis.debug_templates')),
        )

        features = []

        for fs in res.flurstueckList:
            f = gws.base.feature.with_model(self.model)
            f.attributes = dict(uid=fs.uid, fs=fs, geometry=fs.shape)
            f.transform_to(crs)
            f.render_views(templates, user=req.user, **args)
            f.attributes.pop('fs')
            features.append(f)

        return FindFlurstueckResponse(
            features=[gws.props(f, req.user, self) for f in features],
            total=res.total,
        )

    # def api_export(self, req: gws.IWebRequester, p: ExportRequest) -> ExportResponse:
    #     """Export Flurstueck features"""
    #
    #     if not self.exporter:
    #         raise gws.web.error.NotFound()
    #
    #     self._validate_request(req, p)
    #
    #     fp = p.findRequest
    #     fp.projectUid = p.projectUid
    #     fp.locale = p.locale
    #
    #     res = self._fetch(req, fp, self.limit)
    #
    #     if not res.features:
    #         raise gws.NotFoundError()
    #
    #     csv_bytes = self.exporter.export_as_csv(res.features, sorted(int(g) for g in p.groups))
    #
    #     return ExportResponse(content=csv_bytes, mime='text/csv')

    @gws.ext.command.api('alkisPrintFlurstueck')
    def print_flurstueck(self, req: gws.IWebRequester, p: PrintFlurstueckRequest) -> gws.base.printer.StatusResponse:
        """Print Flurstueck features"""

        project = req.require_project(p.projectUid)

        fr = p.findRequest
        fr.projectUid = p.projectUid

        res = self._find_flurstueck(req, fr)
        if not res.flurstueckList:
            raise gws.NotFoundError()

        pr = p.printRequest
        pr.projectUid = p.projectUid
        crs = pr.get('crs') or project.map.bounds.crs

        templates = [
            gws.base.template.locate(self, user=req.user, subject='feature.label'),
        ]

        maps = []

        for fs in res.flurstueckList:
            f = gws.base.feature.with_model(self.model)
            f.attributes = dict(uid=fs.uid, fs=fs, geometry=fs.shape)
            f.transform_to(crs)
            f.render_views(templates, user=req.user)
            f.cssSelector = '.modMarkerFeature'
            mp = gws.base.printer.core.MapParams(pr.maps[0])
            c = f.shape().centroid()
            mp.center = (c.x, c.y)
            mp.planes += [
                gws.base.printer.core.Plane(
                    type=gws.base.printer.core.PlaneType.features,
                    features=[gws.props(f, req.user, self)],
                )
            ]
            maps.append(mp)

        pr.maps = maps
        pr.args = dict(
            flurstueckList=res.flurstueckList,
            withHistory=res.queryOptions.withHistoryDisplay,
            withDebug=bool(self.root.app.developer_option('alkis.debug_templates')),
        )

        job = gws.base.printer.job.start(self.root, pr, req.user)
        return gws.base.printer.job.status(job)

    ##

    def _find_flurstueck(self, req: gws.IWebRequester, p: FindFlurstueckRequest) -> FindFlurstueckResult:

        query = dt.FlurstueckSearchQuery(
            flurnummer=p.flurnummer,
            flurstuecksfolge=p.flurstuecksfolge,
            zaehler=p.zaehler,
            nenner=p.nenner,
            flaecheBis=p.flaecheBis,
            flaecheVon=p.flaecheVon,
            gemarkungCode=p.gemarkungCode,
            gemeindeCode=p.gemeindeCode,
            strasse=p.strasse,
            hausnummer=p.hausnummer,
            personName=p.personName,
            personVorname=p.personVorname,
            uids=p.uids,
        )

        if p.vollNummer:
            self._query_vollnummer(query, p.vollNummer)

        if p.shapes:
            shapes = [gws.base.shape.from_props(s) for s in p.shapes]
            query.shape = shapes[0] if len(shapes) == 1 else shapes[0].union(shapes[1:])

        if p.bblatt:
            query.buchungsblattkennzeichen = p.bblatt.replace(';', ' ').replace(',', ' ').strip().split()

        qo = dt.FlurstueckSearchOptions(
            strasseSearchOptions=self.strasseSearchOptions,
            nameSearchOptions=self.nameSearchOptions,
            buchungsblattSearchOptions=self.buchungsblattSearchOptions,
            withHistorySearch=bool(p.wantHistorySearch),
            withHistoryDisplay=bool(p.wantHistoryDisplay),
            displayThemes=p.displayThemes or [],
        )

        want_eigentuemer = (
                dt.DisplayTheme.eigentuemer in qo.displayThemes
                or any(getattr(query, key) is not None for key in dt.EigentuemerAccessRequired)
        )
        if want_eigentuemer and not self._check_eigentuemer_access(req, p.eigentuemerControlInput):
            qo.displayThemes.remove(dt.DisplayTheme.eigentuemer)
            for key in dt.EigentuemerAccessRequired:
                setattr(query, key, None)

        want_buchung = (
                dt.DisplayTheme.buchung in qo.displayThemes
                or any(getattr(query, key) is not None for key in dt.BuchungAccessRequired)
        )
        if want_buchung and not self._check_buchung_access(req, p.eigentuemerControlInput):
            qo.displayThemes.remove(dt.DisplayTheme.buchung)
            for key in dt.BuchungAccessRequired:
                setattr(query, key, None)

        fs_uids = self.index.find_flurstueck_uids(query, qo)
        total = len(fs_uids)
        fs_uids = fs_uids[:self.limit]

        fs_list = self.index.load_flurstueck(fs_uids, qo)
        if want_eigentuemer:
            self._log_eigentuemer_access(req, p.eigentuemerControlInput, is_ok=True, total=total, fs_uids=fs_uids)

        return FindFlurstueckResult(
            flurstueckList=fs_list,
            total=total,
            query=query,
            queryOptions=qo,
        )

    def _query_vollnummer(self, query: dt.FlurstueckSearchQuery, vn: str):

        if vn.startswith('DE'):
            # search by gml_id
            query.uids = query.uids or []
            query.uids.append(vn)
            return

        if re.match('^[0-9_]+$', vn):
            # search by fs kennzeichen
            query.flurstueckskennzeichen = vn
            return

        # search by a compound fs number
        num = index.fs_parse_vollnummer(vn)
        if num is None:
            raise gws.BadRequestError(f'invalid vollNummer {vn!r}')
        query.update(num)

    ##

    def _check_eigentuemer_access(self, req: gws.IWebRequester, control_input: str):
        if not req.user.can_read(self.eigentuemer):
            return False

        if self.eigentuemer.controlMode and not self._check_eigentuemer_control_input(control_input):
            self._log_eigentuemer_access(req, is_ok=False, control_input=control_input)
            raise gws.ForbiddenError()

        return True

    def _check_buchung_access(self, req: gws.IWebRequester, control_input: str):
        if not req.user.can_read(self.buchung):
            return False
        return True

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

    def _log_eigentuemer_access(self, req: gws.IWebRequester, control_input: str, is_ok: bool, total=None, fs_uids=None):
        if not self.eigentuemer.logTable:
            gws.log.debug(f'_log_eigentuemer_access {is_ok=} no log table!')
            return

        data = {
            'app_name': 'gws',
            'date_time': gws.lib.date.now_iso(),
            'ip': req.env('REMOTE_ADDR', ''),
            'login': req.user.uid,
            'user_name': req.user.displayName,
            'control_input': (control_input or '').strip(),
            'control_result': 1 if is_ok else 0,
            'fs_count': total or 0,
            'fs_ids': ','.join(fs_uids or []),
        }

        with self.index.connect() as conn:
            conn.execute(sa.insert(self.eigentuemer.saTable).values([data]))
            conn.commit()

        gws.log.debug('_log_eigentuemer_access', is_ok, 'ok')

    def _check_eigentuemer_control_input(self, control_input):
        if not self.eigentuemer.controlRules:
            return True

        control_input = (control_input or '').strip()

        for rule in self.eigentuemer.controlRules:
            if re.search(rule, control_input):
                return True

        return False