"""Parser for QField-related capabilities of a QGIS project."""

from typing import Optional, cast

import gws
import gws.base.shape
import gws.gis.source
import gws.lib.datetimex as dtx
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.crs
import gws.plugin.qgis.caps

from . import core


class ProjectProps(gws.Data):
    """Custom project properties as defined by QField."""

    areaOfInterest: str
    """Area of interest as WKT."""
    areaOfInterestCrs: str
    """CRS of the area of interest."""
    baseMapLayer: str
    """Id of the layer used as the base map, if ``baseMapType`` is ``singleLayer``."""
    baseMapTheme: str
    """Map theme used as the base map, if ``baseMapType`` is ``mapTheme``."""
    baseMapTileSize: int
    """Base map tile size."""
    baseMapTilesMaxZoomLevel: int
    """Maximum zoom level of the base map."""
    baseMapTilesMinZoomLevel: int
    """Minimum zoom level of the base map."""
    baseMapType: str
    """Base map type, ``mapTheme`` or ``singleLayer``."""
    createBaseMap: bool
    """Whether a base map is created."""
    digitizingLogsLayer: str
    """Id of the digitizing logs layer."""
    forceAutoPush: bool
    """Whether changes are pushed automatically."""
    forceAutoPushIntervalMins: int
    """Automatic push interval in minutes."""
    forceStamping: bool
    """Whether photos are stamped."""
    geofencingBehavior: int
    """Geofencing behavior."""
    geofencingIsActive: bool
    """Whether geofencing is active."""
    geofencingLayer: str
    """Id of the geofencing layer."""
    geofencingShouldPreventDigitizing: bool
    """Whether geofencing prevents digitizing."""
    mapThemesActiveLayers: dict
    """Active layers of map themes."""
    maximumImageWidthHeight: int
    """Maximum width and height of images."""
    offlineCopyOnlyAoi: bool
    """Whether only features in the area of interest are packaged."""
    stampingDetailsTemplate: str
    """Template for photo stamping."""
    stampingFontStyle: str
    """Font style for photo stamping."""
    stampingHorizontalAlignment: int
    """Horizontal alignment for photo stamping."""
    stampingImageDecoration: str
    """Image decoration for photo stamping."""

    attachmentDirs: list[str]
    """Attachment directories."""
    dataDirs: list[str]
    """Data directories."""
    dirsToCopy: dict
    """Directories to copy, as a dict ``{dirname: bool}``."""


class LayerProps(gws.Data):
    """Custom layer properties as defined by QField."""

    action: str
    """QFieldSync layer action."""
    attachment_naming: dict
    """Naming rules for attachments."""
    attribute_editing_locked_expression: str
    """Expression that locks attribute editing."""
    cloud_action: str
    """Cloud action: ``offline``, ``no_action`` or ``remove``."""
    feature_addition_locked_expression: str
    """Expression that locks feature addition."""
    feature_deletion_locked_expression: str
    """Expression that locks feature deletion."""
    geometry_editing_locked_expression: str
    """Expression that locks geometry editing."""
    is_attribute_editing_locked: bool
    """Whether attribute editing is locked."""
    is_feature_addition_locked: bool
    """Whether feature addition is locked."""
    is_feature_deletion_locked: bool
    """Whether feature deletion is locked."""
    is_geometry_editing_locked: bool
    """Whether geometry editing is locked."""
    photo_naming: dict
    """Naming rules for photos."""
    relationship_maximum_visible: dict
    """Maximum number of visible related features."""
    tracking_distance_requirement_minimum_meters: int
    """Minimum distance between tracked positions, in meters."""
    tracking_erroneous_distance_safeguard_maximum_meters: int
    """Maximum distance of a tracked position before it is considered erroneous, in meters."""
    tracking_measurement_type: int
    """Tracking measurement type."""
    tracking_time_requirement_interval_seconds: int
    """Minimum time between tracked positions, in seconds."""
    value_map_button_interface_threshold: int
    """Threshold for showing value maps as buttons."""


class LayerAction(gws.Enum):
    """What the packager does with a layer."""

    remove = 'remove'
    """Remove the layer from the project."""
    edit = 'edit'
    """Package the layer data for offline editing."""
    baseMap = 'baseMap'
    """Render the layer into the base map."""


class ModelEntry(gws.Data):
    """A model for an offline table, with its GeoPackage layer name."""

    gpName: str
    """Name of the GeoPackage file and layer."""
    tableName: str
    """Database table name."""
    model: gws.DatabaseModel
    """Model for the table."""


class LayerEntry(gws.Data):
    """QField-related information about a QGIS layer."""

    action: LayerAction
    """What to do with the layer."""
    qgisId: str
    """Layer id in the QGIS project."""
    modelEntry: ModelEntry
    """Model entry, for ``edit`` layers."""
    readOnly: bool
    """Whether all editing is locked in QFieldSync."""
    sqlFilter: str
    """SQL filter (subset string) of the layer."""
    dataSourceFileName: str
    """Name of the packaged data file."""
    dataSource: str
    """Data source string in the packaged QGIS project."""
    dataProvider: str
    """Data provider in the packaged QGIS project."""
    sourceLayer: gws.SourceLayer
    """Source layer from the QGIS project."""
    props: LayerProps
    """QFieldSync layer properties."""


class Caps(gws.Data):
    """QField related capabilities extracted from the QGIS project and GWS config."""

    sourceHash: str
    """Hash of the QGIS project source, used to invalidate cached caps."""
    qgisPath: str
    """Path to the QGIS project file, empty if the project is not stored in a file."""
    layerMap: dict[str, LayerEntry]
    """Layer entries by QGIS layer id."""
    modelMap: dict[str, ModelEntry]
    """Model entries by GeoPackage name."""
    copyDirs: list[str]
    """Absolute paths of directories to copy into the package."""
    baseMapLayerIds: list[str]
    """Ids of the layers rendered into the base map."""
    areaOfInterest: Optional[gws.Bounds]
    """Area of interest."""
    copyOnlyAreaOfInterest: bool
    """Whether only features in the area of interest are packaged."""
    projectProps: ProjectProps
    """QFieldSync project properties."""


class Parser:
    """Reads QField-related capabilities from a QGIS project.

    ``parse`` reads the project and layer properties. ``create_models`` and
    ``assign_path_props`` complete the layer entries and are called separately.
    """

    project: core.QfcProject
    """QField project (unused, the project is stored in ``qfcProject``)."""
    caps: Caps
    """Capabilities being built."""
    qgisCaps: gws.plugin.qgis.caps.Caps
    """Capabilities of the QGIS project."""

    def __init__(self, qfc_project: core.QfcProject):
        """Create a parser.

        Args:
            qfc_project: QField project to parse.
        """
        self.qfcProject = qfc_project

    def parse(self) -> Caps:
        """Parse the QGIS project.

        Reads the project properties, the area of interest, the directories to copy, the base map layers and the layer entries.

        Returns:
            Capabilities, also stored in ``self.caps``.
        """
        qp = self.qfcProject.qgisProvider.qgis_project()
        self.qgisCaps = qp.caps()

        self.caps = Caps(
            qgisPath='',
            sourceHash=qp.sourceHash,
            layerMap={},
            modelMap={},
            copyDirs=[],
            baseMapLayerIds=[],
            areaOfInterest=None,
            copyOnlyAreaOfInterest=False,
            projectProps=self.extract_project_props(),
        )

        if self.qfcProject.qgisProvider.store.type == gws.plugin.qgis.project.StoreType.file:
            self.caps.qgisPath = self.qfcProject.qgisProvider.store.path

        self.parse_area_of_interest()
        self.parse_copy_dirs()
        self.parse_base_map()

        self.iter_layers()

        return self.caps

    ##

    def parse_area_of_interest(self):
        """Set the area of interest from the project properties."""
        aoi = self.caps.projectProps.areaOfInterest
        if not aoi:
            return
        crs = self.caps.projectProps.areaOfInterestCrs
        shape = gws.base.shape.from_wkt(aoi, gws.lib.crs.get(crs) or self.qgisCaps.projectCrs)
        self.caps.areaOfInterest = shape.bounds()
        self.caps.copyOnlyAreaOfInterest = self.caps.projectProps.offlineCopyOnlyAoi is True

    def parse_copy_dirs(self):
        """Set the directories to copy into the package.

        Relative paths are resolved against the QGIS project file. Nested directories are dropped.
        """
        raw_dirs = []

        # dirsToCopy is a dict (dirname: bool)
        dc = self.caps.projectProps.dirsToCopy or {}
        for k, v in dc.items():
            if v:
                raw_dirs.append(k)

        # attachmentDirs and dataDirs are lists
        raw_dirs.extend(self.caps.projectProps.attachmentDirs or [])
        raw_dirs.extend(self.caps.projectProps.dataDirs or [])

        abs_dirs = []

        for p in raw_dirs:
            if not p.startswith('/'):
                if not self.caps.qgisPath:
                    gws.log.warning(f'cannot determine an absolute path for {p!r}')
                    continue
                p = gws.lib.osx.abs_path(p, self.caps.qgisPath)
            abs_dirs.append(p)

        unnest_dirs = []

        for p in sorted(abs_dirs):
            if any(p.startswith(d) for d in unnest_dirs):
                continue
            unnest_dirs.append(p)

        self.caps.copyDirs = unnest_dirs

    def parse_base_map(self):
        """Set the base map layer ids from the map theme or the single base map layer."""
        if not self.caps.projectProps.createBaseMap:
            return

        bt = self.caps.projectProps.baseMapType

        if bt == 'mapTheme':
            theme = self.caps.projectProps.baseMapTheme
            if not theme:
                gws.log.warning(f'map theme not defined')
                return

            vp = self.qgisCaps.visibilityPresets.get(theme)
            if not vp:
                gws.log.warning(f'map theme {theme!r} not found')
                return
            self.caps.baseMapLayerIds = vp
            return

        if bt == 'singleLayer':
            uid = self.caps.projectProps.baseMapLayer
            if uid:
                self.caps.baseMapLayerIds = [uid]

    ##

    def iter_layers(self):
        """Create layer entries for all non-group source layers."""
        for sl in gws.gis.source.filter_layers(self.qgisCaps.sourceLayers, is_group=False):
            le = self.layer_entry(sl)
            if le:
                self.caps.layerMap[le.qgisId] = le

    def layer_entry(self, sl: gws.SourceLayer) -> Optional[LayerEntry]:
        """Create a layer entry for a source layer.

        Args:
            sl: Source layer.

        Returns:
            Layer entry, or ``None`` if the layer is left unchanged in the package.
        """
        le = self.layer_entry_2(sl)
        if not le:
            return
        le.qgisId = sl.sourceId
        le.sourceLayer = sl

        return le

    def layer_entry_2(self, sl: gws.SourceLayer) -> Optional[LayerEntry]:
        """Determine the action for a source layer.

        Base map layers get ``baseMap``. Layers with the cloud action ``remove`` get ``remove``.
        Postgres layers with the cloud action ``offline`` get ``edit``, offline layers of other providers get ``remove``.

        Args:
            sl: Source layer.

        Returns:
            Layer entry without the id and source layer, or ``None`` for other layers.
        """
        props = self.extract_layer_props(sl)

        if sl.sourceId in self.caps.baseMapLayerIds:
            return LayerEntry(action=LayerAction.baseMap, props=props)

        # 'offline', 'no_action' or 'remove'
        # for "cable", use "action" instead of "cloud_action"
        act = props.cloud_action

        if act == 'remove':
            return LayerEntry(action=LayerAction.remove, props=props)

        if act == 'offline':
            prov = sl.dataSource.get('provider')
            if prov == 'postgres':
                return self.postgres_layer_entry(sl, props)
            # @TODO support offline for other providers?
            gws.log.warning(f'layer {sl.sourceId!r}: offline editing of {prov!r} not supported')
            return LayerEntry(action=LayerAction.remove, props=props)

    def postgres_layer_entry(self, sl, props: LayerProps) -> LayerEntry:
        """Create a layer entry for an offline Postgres layer.

        Layers without a plain table name (e.g. SQL queries) get the ``remove`` action.

        Args:
            sl: Source layer.
            props: QFieldSync layer properties.

        Returns:
            Layer entry.
        """
        read_only = (
            props.is_attribute_editing_locked
            and props.is_geometry_editing_locked
            and props.is_feature_addition_locked
            and props.is_feature_deletion_locked
        )

        table_name = sl.dataSource.get('table')
        if not table_name or table_name.startswith('(') or table_name.upper().startswith('SELECT '):
            gws.log.warning(f'layer {sl.sourceId!r}: no table name')
            return LayerEntry(action=LayerAction.remove, props=props)

        return LayerEntry(
            action=LayerAction.edit,
            readOnly=read_only,
            sqlFilter=sl.dataSource.get('sql', ''),
            props=props,
        )

    ##

    def extract_project_props(self) -> ProjectProps:
        """Read the QFieldSync project properties.

        Returns:
            Project properties.
        """
        d = {}
        # there are two of them, QFieldSync and libqfieldsync
        d.update(self.qgisCaps.properties.get('qfieldsync', {}))
        d.update(self.qgisCaps.properties.get('QFieldSync', {}))

        t = ProjectProps()
        _dict_to_data(d, t)
        return t

    def extract_layer_props(self, sl: gws.SourceLayer) -> LayerProps:
        """Read the QFieldSync properties of a layer.

        Args:
            sl: Source layer.

        Returns:
            Layer properties.
        """
        d = {}

        for k, v in sl.properties.items():
            if k.startswith('QFieldSync/'):
                d[k.split('/').pop()] = v

        t = LayerProps()
        _dict_to_data(d, t)
        return t

    ##

    def assign_path_props(self):
        """Set the data file name, data source and provider of ``edit`` and ``baseMap`` layers in the package."""
        for le in self.caps.layerMap.values():
            if le.action == LayerAction.edit:
                name = le.modelEntry.gpName
                le.dataSourceFileName = f'{name}.gpkg'
                le.dataSource = f'./{le.dataSourceFileName}|layername={name}'
                if le.sqlFilter:
                    le.dataSource += f'|subset={le.sqlFilter}'
                le.dataProvider = 'ogr'

            if le.action == LayerAction.baseMap:
                u = gws.u.to_uid(le.qgisId)
                le.dataSourceFileName = f'{u}.gpkg'
                le.dataSource = f'./{le.dataSourceFileName}'
                le.dataProvider = 'gdal'

    def create_models(self):
        """Create model entries for all ``edit`` layers."""
        self.caps.modelMap = {}

        for le in self.caps.layerMap.values():
            self.create_model_entry_for_layer(le)

    def create_model_entry_for_layer(self, le: LayerEntry):
        """Assign a model entry to an ``edit`` layer.

        If no model is found, or the layer is editable but the model is not, the layer gets the ``remove`` action.

        Args:
            le: Layer entry.
        """
        if le.action != LayerAction.edit:
            return
        me = self.model_entry_for_source_layer(le.sourceLayer)
        if not me:
            gws.log.warning(f'layer {le.qgisId!r}: no model')
            le.action = LayerAction.remove
            return
        if not le.readOnly and not me.model.isEditable:
            gws.log.warning(f'layer {le.qgisId!r}: table {me.tableName!r} is not editable')
            le.action = LayerAction.remove
            return
        le.modelEntry = me

    def model_entry_for_source_layer(self, sl: gws.SourceLayer) -> Optional[ModelEntry]:
        """Find or create a model entry for the table of a source layer.

        A configured model with the same table name is used if present.
        Otherwise a generic Postgres model is created for the table.

        Args:
            sl: Source layer.

        Returns:
            Model entry, or ``None`` if the layer has no table or the table does not exist.
        """
        table_name = sl.dataSource.get('table')
        if not table_name:
            return

        for model in self.qfcProject.models:
            full_name = model.db.join_table_name('', model.tableName)
            if full_name == model.db.join_table_name('', table_name):
                gp_name = self.gp_name_for_model(full_name)
                if gp_name not in self.caps.modelMap:
                    self.caps.modelMap[gp_name] = ModelEntry(gpName=gp_name, tableName=full_name, model=model)
                return self.caps.modelMap[gp_name]

        db = self.qfcProject.qgisProvider.postgres_provider_from_datasource(sl.dataSource)
        if not db.has_table(table_name):
            gws.log.warning(f'layer {sl.sourceId!r}: table {table_name!r} not found')
            return

        gp_name = self.gp_name_for_model(table_name)

        if gp_name not in self.caps.modelMap:
            model = self.qfcProject.root.create_shared(
                gws.ext.object.model,
                gws.Config(
                    uid=f'qfield_model_{table_name}',
                    type='postgres',
                    # NB: permissions are checked in the public export/import functions above
                    permissions=gws.Config(read=gws.c.PUBLIC, edit=gws.c.PUBLIC),
                    tableName=table_name,
                    isEditable=True,
                    _defaultDb=db,
                ),
            )
            self.caps.modelMap[gp_name] = ModelEntry(gpName=gp_name, tableName=table_name, model=cast(gws.DatabaseModel, model))

        return self.caps.modelMap[gp_name]

    def gp_name_for_model(self, table_name):
        """Return the GeoPackage name for a table.

        Args:
            table_name: Table name, optionally with a schema (``public`` by default).

        Returns:
            Name in the form ``qm_<schema>_<table>``, lowercase.
        """
        if '.' not in table_name:
            table_name = 'public.' + table_name
        return 'qm_' + table_name.replace('.', '_').lower()


##


def _dict_to_data(d: dict, t: gws.Data):
    """Copy dict values into a Data object, converting them to the annotated types."""
    for k, typ in t.__class__.__annotations__.items():
        v = d.get(k)
        if v is None:
            continue
        try:
            if typ is bool:
                # QFieldSync writes flags both as `int` (1/0) and as `bool` (true/false)
                v = v is True or str(v).lower() in ('1', 'true')
            elif typ is int:
                v = int(v)
            elif typ is float:
                v = float(v)
            elif typ is dict:
                v = gws.lib.jsonx.from_string(v)
        except Exception as exc:
            gws.log.warning(f'invalid property value {k!r}={v!r}: {exc}')
            continue
        setattr(t, k, v)
    return t
