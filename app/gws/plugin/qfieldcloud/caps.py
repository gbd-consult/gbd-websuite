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
    areaOfInterestCrs: str
    baseMapLayer: str
    baseMapTheme: str
    baseMapTileSize: int
    baseMapTilesMaxZoomLevel: int
    baseMapTilesMinZoomLevel: int
    baseMapType: str
    createBaseMap: bool
    digitizingLogsLayer: str
    forceAutoPush: bool
    forceAutoPushIntervalMins: int
    forceStamping: bool
    geofencingBehavior: int
    geofencingIsActive: bool
    geofencingLayer: str
    geofencingShouldPreventDigitizing: bool
    mapThemesActiveLayers: dict
    maximumImageWidthHeight: int
    offlineCopyOnlyAoi: bool
    stampingDetailsTemplate: str
    stampingFontStyle: str
    stampingHorizontalAlignment: int
    stampingImageDecoration: str

    attachmentDirs: list[str]
    dataDirs: list[str]
    dirsToCopy: dict


class LayerProps(gws.Data):
    """Custom layer properties as defined by QField."""

    action: str
    attachment_naming: dict
    attribute_editing_locked_expression: str
    cloud_action: str
    feature_addition_locked_expression: str
    feature_deletion_locked_expression: str
    geometry_editing_locked_expression: str
    is_attribute_editing_locked: bool
    is_feature_addition_locked: bool
    is_feature_deletion_locked: bool
    is_geometry_editing_locked: bool
    photo_naming: dict
    relationship_maximum_visible: dict
    tracking_distance_requirement_minimum_meters: int
    tracking_erroneous_distance_safeguard_maximum_meters: int
    tracking_measurement_type: int
    tracking_time_requirement_interval_seconds: int
    value_map_button_interface_threshold: int


class LayerAction(gws.Enum):
    remove = 'remove'
    edit = 'edit'
    baseMap = 'baseMap'


class ModelEntry(gws.Data):
    gpName: str
    tableName: str
    model: gws.DatabaseModel


class LayerEntry(gws.Data):
    action: LayerAction
    qgisId: str
    modelEntry: ModelEntry
    readOnly: bool
    sqlFilter: str
    dataSourceFileName: str
    dataSource: str
    dataProvider: str
    sourceLayer: gws.SourceLayer
    props: LayerProps


class Caps(gws.Data):
    """QField related capabilities extracted from the QGIS project and GWS config."""

    sourceHash: str
    qgisPath: str
    layerMap: dict[str, LayerEntry]
    modelMap: dict[str, ModelEntry]
    copyDirs: list[str]
    baseMapLayerIds: list[str]
    areaOfInterest: Optional[gws.Bounds]
    copyOnlyAreaOfInterest: bool
    projectProps: ProjectProps


class Parser:
    """Read qf-related capabilities from the qgis project."""

    project: core.QfcProject
    caps: Caps
    qgisCaps: gws.plugin.qgis.caps.Caps

    def __init__(self, qfc_project: core.QfcProject):
        self.qfcProject = qfc_project

    def parse(self) -> Caps:
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
        aoi = self.caps.projectProps.areaOfInterest
        if not aoi:
            return
        crs = self.caps.projectProps.areaOfInterestCrs
        shape = gws.base.shape.from_wkt(aoi, gws.lib.crs.get(crs) or self.qgisCaps.projectCrs)
        self.caps.areaOfInterest = shape.bounds()
        self.caps.copyOnlyAreaOfInterest = self.caps.projectProps.offlineCopyOnlyAoi is True

    def parse_copy_dirs(self):
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
        for sl in gws.gis.source.filter_layers(self.qgisCaps.sourceLayers, is_group=False):
            le = self.layer_entry(sl)
            if le:
                self.caps.layerMap[le.qgisId] = le

    def layer_entry(self, sl: gws.SourceLayer) -> Optional[LayerEntry]:
        le = self.layer_entry_2(sl)
        if not le:
            return
        le.qgisId = sl.sourceId
        le.sourceLayer = sl

        return le

    def layer_entry_2(self, sl: gws.SourceLayer) -> Optional[LayerEntry]:
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
        d = {}
        # there are two of them, QFieldSync and libqfieldsync
        d.update(self.qgisCaps.properties.get('qfieldsync', {}))
        d.update(self.qgisCaps.properties.get('QFieldSync', {}))

        t = ProjectProps()
        _dict_to_data(d, t)
        return t

    def extract_layer_props(self, sl: gws.SourceLayer) -> LayerProps:
        d = {}

        for k, v in sl.properties.items():
            if k.startswith('QFieldSync/'):
                d[k.split('/').pop()] = v

        t = LayerProps()
        _dict_to_data(d, t)
        return t

    ##

    def assign_path_props(self):
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
        self.caps.modelMap = {}

        for le in self.caps.layerMap.values():
            self.create_model_entry_for_layer(le)

    def create_model_entry_for_layer(self, le: LayerEntry):
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
        if '.' not in table_name:
            table_name = 'public.' + table_name
        return 'qm_' + table_name.replace('.', '_').lower()


##


def _dict_to_data(d: dict, t: gws.Data):
    for k, typ in t.__class__.__annotations__.items():
        v = d.get(k)
        if v is None:
            continue
        try:
            if typ is bool:
                v = str(v) == '1'
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
