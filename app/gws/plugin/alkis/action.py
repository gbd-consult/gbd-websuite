"""The ``alkis`` action, the backend for the Flurstückssuche (cadastre parcel search)."""

from typing import Optional, cast

import os
import re

import gws
import gws.base.action
import gws.base.feature
import gws.base.model
import gws.base.printer
import gws.base.shape
import gws.base.storage
import gws.config.util
import gws.lib.datetimex
import gws.lib.sa as sa

from .data import index, exporter
from .data import types as dt


class EigentuemerConfig(gws.ConfigWithAccess):
    """Access to the Eigentümer (owner) information."""

    controlMode: bool = False
    """Require a control input before owner data is shown."""
    controlRules: Optional[list[str]]
    """Regular expressions the control input must match."""
    logTable: str = ''
    """Table for logging access to owner data."""


class EigentuemerOptions(gws.Node):
    """Access options for owner (Eigentuemer) data.

    Holds the read permissions for owner data, the control mode and the
    table for logging access to owner data.
    """

    controlMode: bool
    """Require a control input before owner data is shown."""
    controlRules: list[str]
    """Regular expressions the control input must match."""
    logTableName: str
    """Name of the table for logging access to owner data."""
    logTable: Optional[sa.Table]
    """Table for logging access to owner data, set by the action."""

    def configure(self):
        self.controlMode = self.cfg('controlMode')
        self.controlRules = self.cfg('controlRules', default=[])
        self.logTableName = self.cfg('logTable')
        self.logTable = None


class BuchungConfig(gws.ConfigWithAccess):
    """Access to the Grundbuch (register) information."""

    pass


class BuchungOptions(gws.Node):
    """Access options for the Grundbuch (register) data.

    Holds the read permissions for register data.
    """

    pass


class GemarkungListMode(gws.Enum):
    """How the Gemarkung list is shown."""

    none = 'none'
    """Do not show the list."""
    plain = 'plain'
    """Show only the Gemarkung."""
    combined = 'combined'
    """Show 'Gemarkung (Gemeinde)'."""
    tree = 'tree'
    """A tree with level 1 = Gemeinde and level 2 = Gemarkung."""


class StrasseListMode(gws.Enum):
    """Strasse (street) list entry format."""

    plain = 'plain'
    """Show only the street name."""
    withGemeinde = 'withGemeinde'
    """Show 'Strasse (Gemeinde)'."""
    withGemarkung = 'withGemarkung'
    """Show 'Strasse (Gemarkung)'."""
    withGemeindeIfRepeated = 'withGemeindeIfRepeated'
    """Show 'Strasse (Gemeinde)' when needed for disambiguation."""
    withGemarkungIfRepeated = 'withGemarkungIfRepeated'
    """Show 'Strasse (Gemarkung)' when needed for disambiguation."""


class Ui(gws.Config):
    """User interface options for the Flurstückssuche."""

    useExport: bool = False
    """Enable the export of search results."""
    useSelect: bool = False
    """Enable the selection storage for parcels."""
    usePick: bool = False
    """Enable picking single parcels in the map."""
    useHistory: bool = False
    """Enable options to search and display historic parcels."""
    searchSelection: bool = False
    """Enable searching within the geometries of the current map selection."""
    searchSpatial: bool = False
    """Enable spatial search with a search area drawn in the map."""
    gemarkungListMode: GemarkungListMode = GemarkungListMode.combined
    """How the Gemarkung list is shown in the search form."""
    strasseListMode: StrasseListMode = StrasseListMode.plain
    """How entries in the street list are shown."""
    autoSpatialSearch: bool = False
    """Start the spatial search tool after a form search."""


@gws.ext.config.action('alkis')
class Config(gws.ConfigWithAccess):
    """Parcel search in ALKIS data."""

    dbUid: str = ''
    """UID of the PostgreSQL database provider with the ALKIS data."""
    crs: gws.CrsName
    """CRS for the ALKIS data."""
    dataSchema: str = 'public'
    """Schema where ALKIS tables are stored."""
    indexSchema: str = 'gws8'
    """Schema for the GWS search indexes built from the ALKIS data."""
    excludeGemarkung: Optional[list[str]]
    """Gemarkung (Administrative Unit) IDs to exclude from indexing."""
    gemarkungFilter: Optional[list[str]]
    """Gemarkung IDs to restrict the search to."""

    eigentuemer: Optional[EigentuemerConfig]
    """Access to the owner information."""
    buchung: Optional[BuchungConfig]
    """Access to the land register information."""
    limit: int = 100
    """Maximum number of parcels returned by a search."""
    templates: Optional[list[gws.ext.config.template]]
    """Templates for parcel details."""
    printers: Optional[list[gws.base.printer.Config]]
    """Print templates for parcels."""
    ui: Optional[Ui]
    """User interface options for the search form and results."""

    strasseSearchOptions: Optional[gws.TextSearchOptions]
    """How street names are matched."""
    nameSearchOptions: Optional[gws.TextSearchOptions]
    """How person names are matched."""
    buchungsblattSearchOptions: Optional[gws.TextSearchOptions]
    """How book and page numbers are matched."""

    storage: Optional[gws.base.storage.Config]
    """Storage for saved parcel selections."""

    exporters: Optional[list[exporter.Config]]
    """Export formats for parcel data."""


##


@gws.ext.props.action('alkis')
class Props(gws.base.action.Props):
    exporters: list[exporter.Props]
    limit: int
    printer: Optional[gws.base.printer.Props]
    ui: Ui
    storage: Optional[gws.base.storage.Props]
    strasseSearchOptions: Optional[gws.TextSearchOptions]
    nameSearchOptions: Optional[gws.TextSearchOptions]
    buchungsblattSearchOptions: Optional[gws.TextSearchOptions]
    withBuchung: bool
    withEigentuemer: bool
    withEigentuemerControl: bool
    withFlurnummer: bool


##


class GetToponymsRequest(gws.Request):
    """Request for the ``alkisGetToponyms`` command."""

    pass


class GetToponymsResponse(gws.Response):
    """Response of the ``alkisGetToponyms`` command."""

    gemeinde: list[list[str]]
    """Gemeinden (municipalities) as ``[name, code]`` pairs, sorted."""
    gemarkung: list[list[str]]
    """Gemarkungen (districts) as ``[name, code, gemeindeCode]`` lists, sorted."""
    strasse: list[list[str]]
    """Streets as ``[name, gemeindeCode, gemarkungCode]`` lists, sorted."""


class FindFlurstueckRequest(gws.Request):
    """Request for a Flurstück (parcel) search."""

    flurnummer: Optional[str]
    """Flur number."""
    flurstuecksfolge: Optional[str]
    """Flurstücksfolge (parcel sequence number)."""
    zaehler: Optional[str]
    """Numerator of the parcel number."""
    nenner: Optional[str]
    """Denominator of the parcel number."""
    fsnummer: Optional[str]
    """Parcel identifier: a ``DE...`` gml id, a Flurstückskennzeichen or a compound parcel number."""

    flaecheBis: Optional[float]
    """Maximum parcel area."""
    flaecheVon: Optional[float]
    """Minimum parcel area."""

    gemarkung: Optional[str]
    """Gemarkung (district) name."""
    gemarkungCode: Optional[str]
    """Gemarkung code."""
    gemeinde: Optional[str]
    """Gemeinde (municipality) name."""
    gemeindeCode: Optional[str]
    """Gemeinde code."""
    kreis: Optional[str]
    """Kreis (county) name."""
    kreisCode: Optional[str]
    """Kreis code."""
    land: Optional[str]
    """Land (state) name."""
    landCode: Optional[str]
    """Land code."""
    regierungsbezirk: Optional[str]
    """Regierungsbezirk (administrative region) name."""
    regierungsbezirkCode: Optional[str]
    """Regierungsbezirk code."""

    strasse: Optional[str]
    """Street name."""
    hausnummer: Optional[str]
    """House number."""

    bblatt: Optional[str]
    """Buchungsblatt (register sheet) numbers, separated by spaces, commas or semicolons."""

    personName: Optional[str]
    """Last name of the owner."""
    personVorname: Optional[str]
    """First name of the owner."""

    combinedFlurstueckCode: Optional[str]
    """Combined code ``land_gemarkung_flur_zaehler_nenner_folge``; empty and ``0`` parts are ignored."""

    shapes: Optional[list[gws.base.shape.Props]]
    """Search area; several shapes are merged."""

    uids: Optional[list[str]]
    """Parcel uids."""

    crs: Optional[gws.CrsName]
    """CRS for the returned features, defaults to the project map CRS."""
    eigentuemerControlInput: Optional[str]
    """Control input for owner data access in control mode."""
    limit: Optional[int]
    """Unused, the action limit applies."""

    wantEigentuemer: Optional[bool]
    """Request owner data."""
    wantHistorySearch: Optional[bool]
    """Include historic parcels in the search."""
    wantHistoryDisplay: Optional[bool]
    """Include history in the parcel details."""

    displayThemes: Optional[list[dt.DisplayTheme]]
    """Themes to include in the parcel details."""


class FindFlurstueckResponse(gws.Response):
    """Response of a Flurstück (parcel) search."""

    features: list[gws.FeatureProps]
    """Found parcels as features, sorted by title."""
    total: int
    """Number of found parcels."""


class FindFlurstueckResult(gws.Data):
    """Result of a Flurstück (parcel) search."""

    flurstueckList: list[dt.Flurstueck]
    """Found parcels."""
    total: int
    """Number of found parcels."""
    query: dt.FlurstueckQuery
    """The query that was run."""


class FindAdresseRequest(gws.Request):
    """Request for an Adresse (address) search."""

    crs: Optional[gws.CrsName]
    """CRS for the returned features, defaults to the project map CRS."""

    gemarkung: Optional[str]
    """Gemarkung (district) name."""
    gemarkungCode: Optional[str]
    """Gemarkung code."""
    gemeinde: Optional[str]
    """Gemeinde (municipality) name."""
    gemeindeCode: Optional[str]
    """Gemeinde code."""
    kreis: Optional[str]
    """Kreis (county) name."""
    kreisCode: Optional[str]
    """Kreis code."""
    land: Optional[str]
    """Land (state) name."""
    landCode: Optional[str]
    """Land code."""
    regierungsbezirk: Optional[str]
    """Regierungsbezirk (administrative region) name."""
    regierungsbezirkCode: Optional[str]
    """Regierungsbezirk code."""

    strasse: Optional[str]
    """Street name."""
    hausnummer: Optional[str]
    """House number."""
    bisHausnummer: Optional[str]
    """Upper bound of a house number range."""
    hausnummerNotNull: Optional[bool]
    """Return only addresses with a house number."""

    wantHistorySearch: Optional[bool]
    """Include historic addresses in the search."""

    combinedAdresseCode: Optional[str]
    """Combined code ``strasse_hausnummer_plz_gemeinde_bisHausnummer``; empty and ``0`` parts are ignored."""


class FindAdresseResponse(gws.Response):
    """Response of an Adresse (address) search."""

    features: list[gws.FeatureProps]
    """Found addresses as features."""
    total: int
    """Number of found addresses."""


class PrintFlurstueckRequest(gws.Request):
    """Request to print found parcels."""

    findRequest: FindFlurstueckRequest
    """The parcel search to run."""
    printRequest: gws.PrintRequest
    """The print request; its first map is used as the base map for each parcel."""
    featureStyle: gws.StyleProps
    """Style for the parcel features."""


class ExportFlurstueckRequest(gws.Request):
    """Request to export found parcels."""

    findRequest: FindFlurstueckRequest
    """The parcel search to run."""
    exporterUid: str
    """Exporter uid, the first usable exporter if empty."""
    modelUids: Optional[list[str]]
    """Uids of the export models to use."""
    eigentuemerControlInput: Optional[str]
    """Control input for owner data access, unused; the one in ``findRequest`` is checked."""


class ExportFlurstueckResponse(gws.Response):
    """Response of a parcel export."""

    content: str
    """Exported data."""
    mimeType: str
    """Mime type of the exported data."""


##


_dir = os.path.dirname(__file__)

_DEFAULT_TEMPLATES = [
    gws.Config(
        subject='flurstueck.title',
        type='html',
        path=f'{_dir}/templates/flurstueck_title.cx.html',
    ),
    gws.Config(
        subject='flurstueck.teaser',
        type='html',
        path=f'{_dir}/templates/flurstueck_title.cx.html',
    ),
    gws.Config(
        subject='flurstueck.label',
        type='html',
        path=f'{_dir}/templates/flurstueck_label.cx.html',
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
        path=f'{_dir}/templates/adresse_label.cx.html',
    ),
]

_DEFAULT_PRINTER = gws.Config(
    uid='gws.plugin.alkis.default_printer',
    template=gws.Config(
        type='html',
        path=f'{_dir}/templates/print.cx.html',
    ),
    qualityLevels=[gws.TemplateQualityLevel(dpi=72)],
)


##


class Model(gws.base.model.default_model.Object):
    """Model for the parcel and address features returned by the action."""

    def configure(self):
        self.uidName = 'uid'
        self.geometryName = 'geometry'
        self.loadingStrategy = gws.FeatureLoadingStrategy.all


@gws.ext.object.action('alkis')
class Object(gws.base.action.Object):
    """The ``alkis`` action.

    Searches parcels and addresses in the ALKIS index, returns them with
    owner and register data where the user may read it, and prints, exports
    and stores selections of parcels.
    """

    db: gws.DatabaseProvider
    """Database provider with the ALKIS data."""

    ix: index.Object
    """The ALKIS index."""
    ixStatus: dt.IndexStatus
    """Index status, set on activation."""

    buchung: BuchungOptions
    """Access options for register data."""
    eigentuemer: EigentuemerOptions
    """Access options for owner data."""

    dataSchema: str
    """Schema with the ALKIS source tables."""
    indexSchema: str
    """Schema with the index tables."""

    model: gws.Model
    """Model for the returned features."""
    ui: Ui
    """User interface options."""
    limit: int
    """Maximum number of parcels returned by a search."""

    templates: list[gws.Template]
    """Templates for parcel and address views."""
    printers: list[gws.Printer]
    """Printers for parcels."""

    exporters: list[exporter.Object]
    """Exporters for parcel data."""

    strasseSearchOptions: gws.TextSearchOptions
    """How street names are matched."""
    nameSearchOptions: gws.TextSearchOptions
    """How person names are matched."""
    buchungsblattSearchOptions: gws.TextSearchOptions
    """How register sheet numbers are matched."""

    storage: Optional[gws.base.storage.Object]
    """Storage for saved parcel selections."""

    def configure(self):
        gws.config.util.configure_database_provider_for(self, ext_type='postgres')

        self.dataSchema = self.cfg('dataSchema')
        self.indexSchema = self.cfg('indexSchema')

        self.ix = self.root.create(
            index.Object,
            _defaultDb=self.db,
            crs=self.cfg('crs'),
            schema=self.indexSchema,
            excludeGemarkung=self.cfg('excludeGemarkung'),
            gemarkungFilter=self.cfg('gemarkungFilter'),
            uid=f'gws.plugin.alkis.data.index.{self.indexSchema}',
        )

        self.limit = self.cfg('limit')
        self.model = self.create_child(Model)
        self.ui = self.cfg(
            'ui',
            default=Ui(
                gemarkungListMode=GemarkungListMode.combined,
                strasseListMode=StrasseListMode.plain,
            ),
        )

        p = self.cfg('templates', default=[]) + _DEFAULT_TEMPLATES
        self.templates = [self.create_child(gws.ext.object.template, c) for c in p]

        self.printers = self.create_children(gws.ext.object.printer, self.cfg('printers'))
        self.printers.append(self.create_child(gws.ext.object.printer, _DEFAULT_PRINTER))

        d = gws.TextSearchOptions(type=gws.TextSearchType.begin, caseSensitive=False)
        self.strasseSearchOptions = self.cfg('strasseSearchOptions', default=d)

        d = gws.TextSearchOptions(type=gws.TextSearchType.begin, caseSensitive=False)
        self.nameSearchOptions = self.cfg('nameSearchOptions', default=d)

        d = gws.TextSearchOptions(type=gws.TextSearchType.exact, caseSensitive=False)
        self.buchungsblattSearchOptions = self.cfg('buchungsblattSearchOptions', default=d)

        deny_all = gws.Config(access='deny all')

        self.buchung = self.create_child(BuchungOptions, self.cfg('buchung', default=deny_all))

        self.eigentuemer = self.create_child(EigentuemerOptions, self.cfg('eigentuemer', default=deny_all))
        if self.eigentuemer.logTableName:
            self.eigentuemer.logTable = self.ix.db.table(self.eigentuemer.logTableName)

        self.storage = self.create_child_if_configured(gws.base.storage.Object, self.cfg('storage'), categoryName='Alkis')

        self.exporters = []
        p = self.cfg('exporters')
        if p:
            self.exporters = self.create_children(exporter.Object, p)
        elif self.ui.useExport:
            self.exporters.append(self.create_child(exporter.Object))

    def activate(self):
        def _load():
            s = self.ix.status()
            if s.missing:
                self.root.config_warning(f'ALKIS: index not found in schema {self.indexSchema}')
            return s

        self.ixStatus = gws.u.get_server_global(f'gws.plugin.alkis.action.ixStatus.{self.indexSchema}', _load)

    def props(self, user):
        if not self.ixStatus.basic:
            return None

        ps: Props = cast(
            Props,
            gws.u.merge(
                super().props(user),
                limit=self.limit,
                printer=gws.u.first(p for p in self.printers if user.can_use(p)),
                ui=self.ui,
                storage=self.storage,
                withBuchung=(self.ixStatus.buchung and user.can_read(self.buchung)),
                withEigentuemer=(self.ixStatus.eigentuemer and user.can_read(self.eigentuemer)),
                withEigentuemerControl=(self.ixStatus.eigentuemer and user.can_read(self.eigentuemer) and self.eigentuemer.controlMode),
            ),
        )

        ps.exporters = []
        for exp in self.exporters:
            p = exp.props_with_flags(user, withEigentuemer=ps.withEigentuemer, withBuchung=ps.withBuchung)
            if p:
                ps.exporters.append(p)

        ps.strasseSearchOptions = self.strasseSearchOptions
        if ps.withBuchung:
            ps.buchungsblattSearchOptions = self.buchungsblattSearchOptions
        if ps.withEigentuemer:
            ps.nameSearchOptions = self.nameSearchOptions

        return ps

    @gws.ext.command.api('alkisGetToponyms')
    def get_toponyms(self, req: gws.WebRequester, p: GetToponymsRequest) -> GetToponymsResponse:
        """Return all toponyms (Gemeinde, Gemarkung, Strasse) in the index."""

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
            strasse=sorted(strasse_lst),
        )

    @gws.ext.command.api('alkisFindAdresse')
    def find_adresse(self, req: gws.WebRequester, p: FindAdresseRequest) -> FindAdresseResponse:
        """Search for addresses."""

        project = req.user.require_project(p.projectUid)
        crs = p.get('crs') or project.map.bounds.crs

        ad_list, query = self.find_adresse_objects(req, p)
        if not ad_list:
            return FindAdresseResponse(
                features=[],
                total=0,
            )

        templates = [
            self.root.app.templateMgr.find_template('adresse.title', where=[self], user=req.user),
            self.root.app.templateMgr.find_template('adresse.teaser', where=[self], user=req.user),
            self.root.app.templateMgr.find_template('adresse.label', where=[self], user=req.user),
        ]

        fprops = []
        mc = gws.ModelContext(op=gws.ModelOperation.read, target=gws.ModelReadTarget.map, user=req.user)

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
        """Search for parcels."""

        project = req.user.require_project(p.projectUid)
        crs = p.get('crs') or project.map.bounds.crs

        fs_list, query = self.find_flurstueck_objects(req, p)
        if not fs_list:
            return FindFlurstueckResponse(
                features=[],
                total=0,
            )

        templates = [
            self.root.app.templateMgr.find_template('flurstueck.title', where=[self], user=req.user),
            self.root.app.templateMgr.find_template('flurstueck.teaser', where=[self], user=req.user),
        ]

        if query.options.displayThemes:
            templates.append(
                self.root.app.templateMgr.find_template('flurstueck.description', where=[self], user=req.user),
            )

        args = dict(
            withHistory=query.options.withHistoryDisplay,
            withDebug=bool(self.root.app.developer_option('alkis.debug_templates')),
        )

        ps = []
        mc = gws.ModelContext(op=gws.ModelOperation.read, target=gws.ModelReadTarget.map, user=req.user)

        for fs in fs_list:
            f = gws.base.feature.new(model=self.model)
            f.attributes = dict(uid=fs.uid, fs=fs, geometry=fs.shape)
            f.transform_to(crs)
            f.render_views(templates, user=req.user, **args)
            f.attributes.pop('fs')
            ps.append(self.model.feature_to_view_props(f, mc))

        return FindFlurstueckResponse(
            features=sorted(ps, key=lambda p: p.views['title']),
            total=len(fs_list),
        )

    def get_exporter(self, uid: Optional[str], user: gws.User) -> Optional[exporter.Object]:
        """Find an exporter the user can use.

        Args:
            uid: Exporter uid. If empty, the first usable exporter is returned.
            user: The user.

        Returns:
            The exporter, or ``None`` if not found.
        """

        for exp in self.exporters:
            if user.can_use(exp) and (not uid or exp.uid == uid):
                return exp

    @gws.ext.command.api('alkisExportFlurstueck')
    def export_flurstueck(self, req: gws.WebRequester, p: ExportFlurstueckRequest) -> ExportFlurstueckResponse:
        """Search for parcels and export them."""

        exp = self.get_exporter(p.exporterUid, req.user)
        if not exp:
            raise gws.NotFoundError()

        models = exp.get_models(req.user, p.modelUids)
        if not models:
            raise gws.NotFoundError()

        if any(m.withEigentuemer for m in models):
            self._check_eigentuemer_access(req, p.findRequest.eigentuemerControlInput or '')
        if any(m.withBuchung for m in models):
            self._check_buchung_access(req, p.findRequest.eigentuemerControlInput or '')

        find_request = p.findRequest
        find_request.projectUid = p.projectUid

        fs_list, _ = self.find_flurstueck_objects(req, find_request)
        if not fs_list:
            raise gws.NotFoundError()

        args = exporter.Args(
            fsList=fs_list,
            user=req.user,
            models=models,
            path = gws.u.ephemeral_path('alkis_export'),
        )

        exp.run(args)

        return ExportFlurstueckResponse(
            content=gws.u.read_file_b(args.path),
            mimeType=exp.mimeType,
        )

    @gws.ext.command.api('alkisPrintFlurstueck')
    def print_flurstueck(self, req: gws.WebRequester, p: PrintFlurstueckRequest) -> gws.JobStatusResponse:
        """Search for parcels and start a print job for them."""

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
            self.root.app.templateMgr.find_template('flurstueck.label', where=[self], user=req.user),
        ]

        base_map = print_request.maps[0]
        fs_maps = []

        mc = gws.ModelContext(op=gws.ModelOperation.read, target=gws.ModelReadTarget.map, user=req.user)

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

        return self.root.app.printerMgr.start_print_job(print_request, req.user)

    @gws.ext.command.api('alkisSelectionStorage')
    def handle_storage(self, req: gws.WebRequester, p: gws.base.storage.Request) -> gws.base.storage.Response:
        """Read or write saved parcel selections."""

        if not self.storage:
            raise gws.NotFoundError('no storage configured')
        return self.storage.handle_request(req, p)

    ##

    def find_flurstueck_objects(self, req: gws.WebRequester, p: FindFlurstueckRequest) -> tuple[list[dt.Flurstueck], dt.FlurstueckQuery]:
        """Run a parcel search in the index.

        Checks access to owner and register data if requested, and logs owner data access.

        Args:
            req: Web requester.
            p: Search parameters.

        Returns:
            A tuple of the found parcels and the query.

        Raises:
            ``gws.ForbiddenError`` if owner or register data is requested without access.
            ``gws.BadRequestError`` if the parcel number is invalid.
        """

        query = self._prepare_flurstueck_query(req, p)
        fs_list = self.ix.find_flurstueck(query)

        if query.options.withEigentuemer:
            self._log_eigentuemer_access(
                req,
                p.eigentuemerControlInput or '',
                is_ok=True,
                total=len(fs_list),
                fs_uids=[fs.uid for fs in fs_list],
            )

        return fs_list, query

    def find_adresse_objects(self, req: gws.WebRequester, p: FindAdresseRequest) -> tuple[list[dt.Adresse], dt.AdresseQuery]:
        """Run an address search in the index.

        Args:
            req: Web requester.
            p: Search parameters.

        Returns:
            A tuple of the found addresses and the query.
        """

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
    """Request fields copied into a parcel query."""
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
    """Request fields copied into an address query."""
    COMBINED_FLURSTUECK_FIELDS = ['landCode', 'gemarkungCode', 'flurnummer', 'zaehler', 'nenner', 'flurstuecksfolge']
    """Query fields for the parts of a combined parcel code, in order."""
    COMBINED_ADRESSE_FIELDS = ['strasse', 'hausnummer', 'plz', 'gemeinde', 'bisHausnummer']
    """Query fields for the parts of a combined address code, in order."""

    def _prepare_flurstueck_query(self, req: gws.WebRequester, p: FindFlurstueckRequest) -> dt.FlurstueckQuery:
        """Build a parcel query from a request and check access to owner and register data."""

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
            limit=self.limit,
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
            self._check_eigentuemer_access(req, p.eigentuemerControlInput or '')
            options.withEigentuemer = True

        want_buchung = dt.DisplayTheme.buchung in options.displayThemes or any(getattr(query, f) is not None for f in dt.BuchungAccessRequired)
        if want_buchung:
            self._check_buchung_access(req, p.eigentuemerControlInput or '')
            options.withBuchung = True

        # "eigentuemer" implies "buchung"
        if want_eigentuemer and not want_buchung:
            options.withBuchung = True
            options.displayThemes.append(dt.DisplayTheme.buchung)

        query.options = options
        return query

    def _prepare_adresse_query(self, req: gws.WebRequester, p: FindAdresseRequest) -> dt.AdresseQuery:
        """Build an address query from a request."""

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
        """Add a parcel identifier (gml id, Flurstueckskennzeichen or compound number) to the query."""

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
        """Set query fields from the underscore-separated parts of a combined code."""

        for val, field in zip(code_value.split('_'), code_fields):
            if val and val != '0':
                setattr(query, field, val)

    ##

    def _check_eigentuemer_access(self, req: gws.WebRequester, control_input: str):
        """Raise ``gws.ForbiddenError`` if the user may not read owner data or the control input fails."""

        if not req.user.can_read(self.eigentuemer):
            raise gws.ForbiddenError('cannot read eigentuemer')
        if self.eigentuemer.controlMode and not self._check_eigentuemer_control_input(control_input):
            self._log_eigentuemer_access(req, is_ok=False, control_input=control_input)
            raise gws.ForbiddenError('eigentuemer control input failed')

    def _check_buchung_access(self, req: gws.WebRequester, control_input: str):
        """Raise ``gws.ForbiddenError`` if the user may not read register data."""

        if not req.user.can_read(self.buchung):
            raise gws.ForbiddenError('cannot read buchung')

    def _log_eigentuemer_access(self, req: gws.WebRequester, control_input: str, is_ok: bool, total=None, fs_uids=None):
        """Write an owner data access record to the log table, if configured."""

        if self.eigentuemer.logTable is None:
            return

        data = dict(
            app_name='gws',
            date_time=gws.lib.datetimex.now(),
            ip=req.ip,
            login=req.user.uid,
            user_name=req.user.displayName,
            control_input=(control_input or '').strip(),
            control_result=1 if is_ok else 0,
            fs_count=total or 0,
            fs_ids=','.join(fs_uids or []),
        )

        with self.ix.db.connect() as conn:
            conn.execute(sa.insert(self.eigentuemer.logTable).values([data]))
            conn.commit()

        gws.log.debug(f'alkis: _log_eigentuemer_access {is_ok=}')

    def _check_eigentuemer_control_input(self, control_input):
        """Return ``True`` if there are no control rules or the input matches one of them."""

        if not self.eigentuemer.controlRules:
            return True

        control_input = (control_input or '').strip()

        for rule in self.eigentuemer.controlRules:
            if re.search(rule, control_input):
                return True

        return False
