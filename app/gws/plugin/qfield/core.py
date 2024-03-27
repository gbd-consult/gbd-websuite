"""Qfield reader and writer."""

import shutil

import gws
import gws.base.database
import gws.gis.crs
import gws.gis.extent
import gws.gis.gdalx
import gws.gis.render
import gws.gis.source
import gws.lib.image
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.sa as sa
import gws.plugin.model_field.file
import gws.plugin.qgis
import gws.types as t

GPKG_EXT = 'gpkg'


class PackageConfig(gws.ConfigWithAccess):
    qgisProvider: gws.plugin.qgis.provider.Config
    """QGis provider settings"""
    models: t.Optional[list[gws.ext.config.model]]
    """data models"""


class Package(gws.Node):
    qgisProvider: gws.plugin.qgis.provider.Object
    models: list[gws.IDatabaseModel]

    def configure(self):
        self.qgisProvider = self.create_child(gws.plugin.qgis.provider.Object, self.cfg('qgisProvider'))
        self.models = self.create_children(gws.ext.object.model, self.cfg('models'))


class ExportArgs(gws.Data):
    package: Package
    project: gws.IProject
    user: gws.IUser
    baseDir: str
    qgisFileName: str
    dbFileName: str
    dbPath: str
    withBaseMap: bool
    withData: bool
    withMedia: bool
    withQgis: bool


class ImportArgs(gws.Data):
    package: Package
    project: gws.IProject
    user: gws.IUser
    baseDir: str
    dbFileName: str


class LayerAction(t.Enum):
    remove = 'remove'
    edit = 'edit'
    baseMap = 'baseMap'


class EditAction(gws.Enum):
    update = 'update'
    geometryUpdate = 'geometryUpdate'
    insert = 'insert'
    delete = 'delete'


class EditOperation(gws.Data):
    action: EditAction
    fid: int
    pkey: str
    columnName: str
    attributes: dict


class ModelEntry(gws.Data):
    gpId: int
    gpName: str
    model: gws.IDatabaseModel
    fidToPkey: dict
    columnIndex: dict
    editOperations: list[EditOperation]
    features: list[gws.IFeature]


class LayerEntry(gws.Data):
    action: LayerAction
    qgisId: str
    modelEntry: ModelEntry
    readOnly: bool
    sqlFilter: str
    dataSourcePath: str
    dataSource: str
    dataProvider: str
    sourceLayer: gws.SourceLayer


class QFieldCaps(gws.Data):
    qgisPath: str
    layerMap: dict[str, LayerEntry]
    modelMap: dict[str, ModelEntry]
    globalProps: dict
    dirsToCopy: list[str]
    baseMapLayerIds: list[str]


class OfflineLog(gws.Data):
    log_added_attrs: list[dict]
    log_added_features: list[dict]
    log_feature_updates: list[dict]
    log_fids: list[dict]
    log_geometry_updates: list[dict]
    log_indices: list[dict]
    log_layer_ids: list[dict]
    log_removed_features: list[dict]
    log_gws_columns: list[dict]
    log_gws_tables: list[dict]


_GP_ATTRIBUTE_TYPES = {
    gws.AttributeType.bool,
    gws.AttributeType.date,
    gws.AttributeType.datetime,
    gws.AttributeType.float,
    gws.AttributeType.int,
    gws.AttributeType.str,
    gws.AttributeType.time,
}


class Exporter:
    package: Package
    project: gws.IProject
    user: gws.IUser
    args: ExportArgs

    sourceQgisProject: gws.plugin.qgis.project.Object

    targetQgisPath: str

    deviceDbPath: str
    localDbPath: str

    qfCaps: QFieldCaps

    def run(self, args: ExportArgs):
        self.prepare(args)

        if self.args.withData:
            self.write_data()

        if self.args.withBaseMap:
            self.write_base_map()

        if self.args.withMedia:
            self.write_media()

        if self.args.withQgis:
            self.write_qgis_project()

    def prepare(self, args: ExportArgs):
        self.args = args
        self.package = self.args.package
        self.project = self.args.project
        self.user = self.args.user

        self.sourceQgisProject = self.package.qgisProvider.qgis_project()

        self.targetQgisPath = f'{self.args.baseDir}/{self.args.qgisFileName or self.package.uid}.qgs'

        self.localDbPath = f'{self.args.baseDir}/{args.dbPath}'
        self.deviceDbPath = f'./{args.dbPath}'

        self.qfCaps = QFieldCapsParser().run(self.package)

        for le in self.qfCaps.layerMap.values():
            if le.action == LayerAction.edit:
                le.dataSourcePath = self.localDbPath
                le.dataSource = f'{self.deviceDbPath}|layername={le.modelEntry.gpName}'
                if le.sqlFilter:
                    le.dataSource += f'|subset={le.sqlFilter}'
                le.dataProvider = f'ogr'
            if le.action == LayerAction.baseMap:
                file_name = f'{le.qgisId}.{GPKG_EXT}'
                le.dataSourcePath = f'{self.args.baseDir}/{file_name}'
                le.dataSource = f'./{file_name}'
                le.dataProvider = f'gdal'

    def write_data(self):
        with gws.gis.gdalx.open(self.localDbPath, 'w', as_vector=True) as ds:
            for me in self.qfCaps.modelMap.values():
                self.write_features(me, ds)
        self.write_offline_log()

    def write_base_map(self):
        # @TODO options for flattened base maps
        for le in self.qfCaps.layerMap.values():
            if le.action == LayerAction.baseMap:
                self.write_base_map_layer(le)

    def write_media(self):
        for d in self.qfCaps.dirsToCopy:
            if self.qfCaps.qgisPath:
                rel_dir = gws.lib.osx.rel_path(d, self.qfCaps.qgisPath)
            else:
                # @TODO absolute dir with a postgres-based qgis project?
                rel_dir = d.split('/')[-1]
            shutil.copytree(d, f'{self.args.baseDir}/{rel_dir}')

    def write_features(self, me: ModelEntry, ds: gws.gis.gdalx.DataSet):
        # see qgis/src/core/qgsofflineediting.cpp convertToOfflineLayer()

        gp_fields = [f for f in me.model.fields if f.attributeType in _GP_ATTRIBUTE_TYPES]

        gp_layer = ds.create_layer(
            me.gpName,
            columns={f.name: f.attributeType for f in gp_fields},
            geometry_type=me.model.geometryType,
            crs=me.model.geometryCrs,
            overwrite=True
        )

        mc = gws.ModelContext(user=self.user, project=self.project, op=gws.ModelOperation.read)
        features = me.features or me.model.find_features(gws.SearchQuery(), mc)
        records = []

        for feature in features:
            props = me.model.feature_to_props(feature, mc)
            records.append(gws.FeatureRecord(
                attributes={f.name: props.attributes.get(f.name) for f in gp_fields},
                shape=feature.shape(),
                meta={}
            ))

        with ds.transaction():
            fids = gp_layer.insert(records)

        me.columnIndex = {}
        col_idx = 1  # column 0 == fid
        for field in gp_fields:
            me.columnIndex[col_idx] = field.name
            col_idx += 1

        me.fidToPkey = {}
        if me.model.uidName:
            for rec, fid in zip(records, fids):
                me.fidToPkey[fid] = rec.attributes.get(me.model.uidName)

        gws.log.debug(f'{self.args.baseDir}: write_features: {self.package.uid}::{me.gpName!r} count={gp_layer.count()}')

    def write_base_map_layer(self, le: LayerEntry):
        bounds = self.package.qgisProvider.bounds
        resolution = int(self.qfCaps.globalProps.get('baseMapMupp', 10))
        w, h = gws.gis.extent.size(bounds.extent)
        px_size = (w / resolution, h / resolution, gws.Uom.px)

        flat_layer = t.cast(gws.ILayer, self.package.root.create_temporary(
            gws.ext.object.layer,
            type='qgisflat',
            _parentBounds=bounds,
            _parentResolutions=[1],
            _defaultProvider=self.package.qgisProvider,
            _defaultSourceLayers=[le.sourceLayer],
        ))

        mv = gws.gis.render.map_view_from_bbox(
            size=px_size,
            bbox=bounds.extent,
            crs=bounds.crs,
            dpi=96,
            rotation=0,
        )

        lri = gws.LayerRenderInput(
            type=gws.LayerRenderInputType.box,
            user=self.user,
            view=mv,
        )

        lro = flat_layer.render(lri)
        img = gws.lib.image.from_bytes(lro.content)

        src = gws.gis.gdalx.open_image(img, bounds)
        gws.gis.gdalx.create_copy(le.dataSourcePath, src)

    def write_offline_log(self):

        ol = OfflineLog()

        ol.log_fids = [
            {'layer_id': me.gpId, 'offline_fid': fid, 'remote_fid': fid, 'remote_pk': str(pk)}
            for me in self.qfCaps.modelMap.values()
            for fid, pk in me.fidToPkey.items()
        ]
        ol.log_indices = [
            {'name': 'commit_no', 'last_index': 0},
            {'name': 'layer_id', 'last_index': len(self.qfCaps.modelMap)}
        ]

        ol.log_layer_ids = []
        for le in self.qfCaps.layerMap.values():
            if le.action == LayerAction.edit:
                ol.log_layer_ids.append({'id': le.modelEntry.gpId, 'qgis_id': le.qgisId})

        ol.log_gws_columns = []
        for me in self.qfCaps.modelMap.values():
            for col_id, col_name in me.columnIndex.items():
                ol.log_gws_columns.append({'layer_id': me.gpId, 'attr': col_id, 'name': col_name})

        ol.log_gws_tables = [
            {'layer_id': me.gpId, 'name': me.gpName}
            for me in self.qfCaps.modelMap.values()
        ]

        engine = sa.create_engine(f'sqlite:///{self.localDbPath}', echo=False, future=True)
        with engine.begin() as conn:
            _write_offline_log(ol, conn)

    def write_qgis_project(self):
        root_el = self.sourceQgisProject.xml_root()
        QgisXmlTransformer().run(self, root_el)
        xml = root_el.to_string()
        xml = self.replace_vars(xml)
        gws.write_file(self.targetQgisPath, xml)

    def replace_vars(self, s: str) -> str:
        # @TODO render attributes as templates
        s = s.replace('{user.authToken}', self.user.authToken)
        s = s.replace('{user.loginName}', self.user.loginName)
        s = s.replace('{user.displayName}', self.user.displayName)
        return s


class QgisXmlTransformer:
    ex: Exporter
    root: gws.IXmlElement
    remove: list[gws.IXmlElement]

    def run(self, ex: Exporter, root_el: gws.IXmlElement):
        self.ex = ex
        self.root = root_el
        self.remove = []

        self.change_global_props()
        self.update_layer_tree()
        self.update_map_layers()
        self.update_referenced_layers()
        self.update_referenced_layers()
        self.update_edit_widgets()

        self.cleanup_layer_group(root_el.find('layer-tree-group'))
        self.remove_elements(root_el, None)

    def change_global_props(self):
        # change global properties

        properties = self.root.find('properties') or self.root.add('properties')

        # this is added by the Sync plugin
        p = properties.add('OfflineEditingPlugin').add('OfflineDbPath', type='QString')
        p.text = f'{self.ex.deviceDbPath}'

        # ensure relative paths
        p = properties.find('Paths/Absolute')
        if not p:
            p = properties.add('Paths').add('Absolute', type='bool')
        p.text = 'false'

    def update_layer_tree(self):
        for el in self.root.findall('.//layer-tree-layer'):
            le = self.ex.qfCaps.layerMap.get(el.get('id'))
            if not le:
                continue

            if le.action == LayerAction.remove:
                self.remove.append(el)
                continue

            el.set('source', le.dataSource)
            el.set('providerKey', le.dataProvider)

    def update_map_layers(self):
        for el in self.root.findall('.//maplayer'):
            le = self.ex.qfCaps.layerMap.get(el.textof('id'))
            if not le:
                continue

            if le.action == LayerAction.remove:
                self.remove.append(el)
                continue

            el.find('datasource').text = le.dataSource
            el.find('provider').text = le.dataProvider

            if le.action == LayerAction.edit:
                el.remove(el.find('customproperties'))
                opt = el.add('customproperties').add('Option', type='Map')

                opt.add('Option', type='QString', name='QFieldSync/action', value='offline')
                opt.add('Option', type='QString', name='QFieldSync/attachment_naming', value='{}')
                opt.add('Option', type='QString', name='QFieldSync/photo_naming', value='{}')
                opt.add('Option', type='QString', name='QFieldSync/sourceDataPrimaryKeys', value='fid')

                if le.action == LayerAction.edit:
                    if le.readOnly:
                        opt.add('Option', type='bool', name='QFieldSync/is_geometry_locked', value='true')
                    else:
                        opt.add('Option', type='bool', name='isOfflineEditable', value='true')

    def update_referenced_layers(self):
        for el in self.root.findall('.//referencedLayers/relation'):
            """
                <referencedLayers>
                    <relation 
                        strength="Association" 
                        referencingLayer="..."
                        layerId="..."
                        referencedLayer="REF_ID" 
                        providerKey="<REPLACE THIS>"
                        dataSource="<REPLACE THIS>"
                    >
                    ...
            """

            ref_id = el.get('referencedLayer')

            le = self.ex.qfCaps.layerMap.get(ref_id)
            if not le or le.action != LayerAction.edit:
                gws.log.warning(f'relation: referenced layer not found: {ref_id!r}')
                continue

            if 'dataSource' in el.attrib:
                el.set('dataSource', le.dataSource)
            if 'providerKey' in el.attrib:
                el.set('providerKey', le.dataProvider)

    def update_edit_widgets(self):
        for el in self.root.findall('.//editWidget'):
            """
              <editWidget type="RelationReference">
                <config>
                  <Option type="Map">
                    ...
                    <Option value="<REF_ID>" name="ReferencedLayerId" type="QString"/>
                    <Option value="<REPLACE THIS>" name="ReferencedLayerDataSource" type="QString"/>
                    <Option value="<REPLACE THIS>" name="ReferencedLayerProviderKey" type="QString"/>
                    ...
            """

            if el.get('type') != 'RelationReference':
                continue

            ref_id = None
            for opt in el.findall('.//Option'):
                if opt.get('name') == 'ReferencedLayerId':
                    ref_id = opt.get('value')
                    break

            if not ref_id:
                continue

            le = self.ex.qfCaps.layerMap.get(ref_id)
            if not le or le.action != LayerAction.edit:
                gws.log.warning(f'editWidget: referenced layer not found: {ref_id!r}')
                continue

            for opt in el.findall('.//Option'):
                if opt.get('name') == 'ReferencedLayerDataSource':
                    opt.set('value', le.dataSource)
                if opt.get('name') == 'ReferencedLayerProviderKey':
                    opt.set('value', le.dataProvider)

    def cleanup_layer_group(self, group_el):
        is_empty = True

        for sub in group_el.children():
            if sub.tag == 'layer-tree-group':
                if self.cleanup_layer_group(sub):
                    is_empty = False
            if sub.tag == 'layer-tree-layer' and sub not in self.remove:
                is_empty = False

        if is_empty:
            self.remove.append(group_el)

        return not is_empty

    def remove_elements(self, el, parent_el):
        if el in self.remove:
            parent_el.remove(el)
            return
        ns = el.children()
        for n in ns:
            self.remove_elements(n, el)


##

class Importer:
    package: Package
    project: gws.IProject
    user: gws.IUser
    args: ImportArgs

    localDbPath: str
    localImagePaths: list[str]

    updatedModels: list[ModelEntry]

    qfCaps: QFieldCaps

    def run(self, args: ImportArgs):
        self.prepare(args)
        if self.localDbPath:
            self.read_data()

    def prepare(self, args: ImportArgs):
        self.args = args
        self.package = self.args.package
        self.project = self.args.project
        self.user = self.args.user

        self.localDbPath = ''
        self.localImagePaths = []

        for path in gws.lib.osx.find_files(self.args.baseDir):
            if path.endswith(args.dbFileName):
                self.localDbPath = path
            if path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                self.localImagePaths.append(path)

        gws.log.debug(f'{self.localDbPath=} {self.localImagePaths=}')
        self.qfCaps = QFieldCapsParser().run(self.package)

        self.updatedModels = []

    def read_data(self):
        self.enumerate_updated_models()
        self.read_features()
        self.read_images()
        self.commit_edits()

    def enumerate_updated_models(self):
        engine = sa.create_engine(f'sqlite:///{self.localDbPath}', echo=False, future=True)

        with engine.begin() as conn:
            ol = _read_offline_log(conn)

        for rec in ol.log_gws_tables:
            gp_name = rec.get('name')

            me = self.qfCaps.modelMap.get(gp_name)
            if not me:
                gws.log.warning(f'offline model {gp_name!r}: not found')
                continue

            me.gpId = rec.get('layer_id')
            has_updates = self.read_offline_log_for_model(me, ol)
            if has_updates:
                self.updatedModels.append(me)

    def read_features(self):
        with gws.gis.gdalx.open(self.localDbPath, 'r', as_vector=True) as ds:
            for me in self.updatedModels:
                self.read_features_for_model(me, ds)

    def read_images(self):
        for me in self.updatedModels:
            self.read_images_for_model(me)

    def read_offline_log_for_model(self, me: ModelEntry, ol: OfflineLog) -> bool:

        me.columnIndex = {}
        for rec in ol.log_gws_columns:
            if rec['layer_id'] == me.gpId:
                me.columnIndex[rec.get('attr')] = rec.get('name')

        me.fidToPkey = {}
        for rec in ol.log_fids:
            if rec['layer_id'] == me.gpId:
                me.fidToPkey[rec.get('offline_fid')] = rec.get('remote_pk')

        me.editOperations = []
        self.read_edit_operations_for_model(me, ol)

        return len(me.editOperations) > 0

    def read_edit_operations_for_model(self, me: ModelEntry, ol: OfflineLog):
        # NB the order is inserts -> updates (sorted by commit_no) -> deletes

        for rec in ol.log_added_features:
            if rec['layer_id'] == me.gpId:
                fid = rec.get('fid')
                me.editOperations.append(EditOperation(action=EditAction.insert, fid=fid))

        ops = {}
        for rec in ol.log_feature_updates:
            if rec['layer_id'] == me.gpId:
                fid = rec.get('fid')
                pkey = me.fidToPkey.get(fid)
                cname = me.columnIndex.get(rec.get('attr'))
                if not pkey or not cname:
                    gws.log.warning(f'invalid update record {rec!r} {pkey=} {cname=}')
                    continue
                ops[pkey, cname] = EditOperation(action=EditAction.update, fid=fid, pkey=pkey, columnName=cname)
        me.editOperations.extend(ops.values())

        ops = {}
        for rec in ol.log_geometry_updates:
            if rec['layer_id'] == me.gpId:
                fid = rec.get('fid')
                pkey = me.fidToPkey.get(fid)
                if not pkey:
                    gws.log.warning(f'invalid update record {rec!r}')
                    continue
                ops[pkey] = EditOperation(action=EditAction.geometryUpdate, fid=fid, pkey=pkey)
        me.editOperations.extend(ops.values())

        for rec in ol.log_removed_features:
            if rec['layer_id'] == me.gpId:
                pkey = me.fidToPkey.get(rec.get('fid'))
                if not pkey:
                    gws.log.warning(f'invalid delete record {rec!r}')
                    continue
                me.editOperations.append(EditOperation(action=EditAction.delete, pkey=pkey))

    def read_features_for_model(self, me: ModelEntry, ds: gws.gis.gdalx.DataSet):

        gp_layer = ds.layer(me.gpId)

        for eo in me.editOperations:
            eo.attributes = {me.model.uidName: eo.pkey}

            if eo.action == EditAction.delete:
                continue

            feature_rec = gp_layer.get_one(eo.fid, encoding='utf8')
            if not feature_rec:
                gws.log.warning(f'{self.args.baseDir}: operation {eo!r}: fid not found')
                continue

            if eo.action == EditAction.update:
                eo.attributes[eo.columnName] = feature_rec.attributes.get(eo.columnName)
                continue

            if eo.action == EditAction.geometryUpdate:
                if not me.model.geometryName:
                    gws.log.warning(f'{self.args.baseDir}: operation {eo!r}: geometry not defined')
                    continue
                if not feature_rec.shape:
                    gws.log.warning(f'{self.args.baseDir}: operation {eo!r}: geometry not found')
                    continue
                eo.attributes[me.model.geometryName] = feature_rec.shape
                continue

            if eo.action == EditAction.insert:
                eo.attributes.update(feature_rec.attributes)
                if me.model.geometryName and feature_rec.shape:
                    eo.attributes[me.model.geometryName] = feature_rec.shape
                # remove primary key if it is null
                if eo.attributes.get(me.model.uidName) is None:
                    eo.attributes.pop(me.model.uidName, None)
                continue

    def read_images_for_model(self, me: ModelEntry):
        file_field_map = {}

        for field in me.model.fields:
            if field.extType == 'file':
                col_name = t.cast(gws.plugin.model_field.file.Object, field).cols.name.name
                file_field_map[col_name] = field

        if not file_field_map:
            return

        for eo in me.editOperations:
            eo.attributes = self.update_file_attributes(eo, file_field_map)

    def update_file_attributes(self, eo: EditOperation, file_field_map):
        atts = {}

        for name, val in eo.attributes.items():
            field = file_field_map.get(name)
            if not field:
                atts[name] = val
                continue

            fv = self.file_value_for_path(val)
            if not fv:
                gws.log.warning(f'file not found: {val!r}')
                continue

            atts[field.name] = fv

        return atts

    def file_value_for_path(self, path):
        file_name = gws.lib.osx.file_name(path)
        for p in self.localImagePaths:
            if gws.lib.osx.file_name(p) == file_name:
                return gws.plugin.model_field.file.FileValue(
                    content=gws.read_file_b(p),
                    name=file_name,
                    path=path,
                    size=gws.lib.osx.file_size(p),
                )

    def commit_edits(self):
        for me in self.updatedModels:
            self.commit_edits_for_model(me)

    def commit_edits_for_model(self, me: ModelEntry):
        for eo in me.editOperations:
            gws.log.debug(f'edit: {self.localDbPath}: {me.gpName=}: {eo}')

            if eo.action in {EditAction.update, EditAction.geometryUpdate}:
                mc = gws.ModelContext(user=self.user, op=gws.ModelOperation.update)
                model = self.check_model(me, self.user, gws.Access.write)
                feature = model.feature_from_props(gws.FeatureProps(attributes=eo.attributes), mc)
                if not model.validate_feature(feature, mc):
                    gws.log.info(f'validation errors: {feature.errors}')
                    raise gws.BadRequestError('Validation error')
                model.update_feature(feature, mc)

            if eo.action == EditAction.insert:
                mc = gws.ModelContext(user=self.user, op=gws.ModelOperation.create)
                model = self.check_model(me, self.user, gws.Access.create)
                feature = model.feature_from_props(gws.FeatureProps(attributes=eo.attributes), mc)
                if not model.validate_feature(feature, mc):
                    gws.log.info(f'validation errors: {feature.errors}')
                    raise gws.BadRequestError('Validation error')
                model.create_feature(feature, mc)

            if eo.action == EditAction.delete:
                mc = gws.ModelContext(user=self.user, op=gws.ModelOperation.delete)
                model = self.check_model(me, self.user, gws.Access.delete)
                feature = model.feature_from_props(gws.FeatureProps(attributes=eo.attributes), mc)
                model.delete_feature(feature, mc)

    def check_model(self, me: ModelEntry, user: gws.IUser, access: gws.Access) -> gws.IModel:
        if not me.model:
            raise gws.ForbiddenError(f'{me.gpName}: model: not found, {access=} {user=}')
        if not me.model.isEditable:
            raise gws.ForbiddenError(f'{me.gpName}: model not editable, {access=} {user=}')
        if not user.can(access, me.model):
            raise gws.ForbiddenError(f'{me.gpName}: model forbidden, {access=} {user=}')
        return me.model


class QFieldCapsParser:
    """Read qf-related capabilities from the qgis project."""

    package: Package
    caps: QFieldCaps
    qgisCaps: gws.plugin.qgis.caps.Caps

    def run(self, package: Package) -> QFieldCaps:
        self.package = package
        self.caps = QFieldCaps(
            layerMap={},
            modelMap={},
            globalProps={},
        )
        self.qgisCaps = package.qgisProvider.qgis_project().caps()

        # for some reason, there are two of them
        self.caps.globalProps.update(self.qgisCaps.properties.get('qfieldsync', {}))
        self.caps.globalProps.update(self.qgisCaps.properties.get('QFieldSync', {}))

        self.caps.qgisPath = ''
        if self.package.qgisProvider.store.type == gws.plugin.qgis.project.StoreType.file:
            self.caps.qgisPath = self.package.qgisProvider.store.path

        self.caps.dirsToCopy = self.check_dirs_to_copy()
        self.caps.baseMapLayerIds = self.base_map_layer_ids()

        for sl in gws.gis.source.filter_layers(self.qgisCaps.sourceLayers, is_group=False):
            le = self.layer_entry(sl)
            if le:
                le.qgisId = sl.sourceId
                le.gpId = len(self.caps.layerMap)
                le.sourceLayer = sl
                self.caps.layerMap[le.qgisId] = le

        return self.caps

    def check_dirs_to_copy(self):
        dc_str = self.caps.globalProps.get('dirsToCopy')
        if not dc_str:
            return []

        try:
            dc = gws.lib.jsonx.from_string(dc_str)
        except gws.lib.jsonx.Error:
            gws.log.warning(f'dirsToCopy: invalid JSON')
            return []

        dirs = []

        for dir_name, flag in sorted(dc.items()):
            if flag is not True:
                continue

            if not dir_name.startswith('/'):
                if not self.caps.qgisPath:
                    gws.log.warning(f'dirsToCopy: cannot determine an absolute path for {dir_name!r}')
                    continue
                dir_name = gws.lib.osx.abs_path(dir_name, self.caps.qgisPath)

            if any(dir_name.startswith(d) for d in dirs):
                continue
            dirs.append(dir_name)

        return dirs

    def base_map_layer_ids(self) -> list[str]:
        f = self.caps.globalProps.get('createBaseMap')
        if f != 1:
            return []

        bt = self.caps.globalProps.get('baseMapType')

        if bt == 'mapTheme':
            th = self.caps.globalProps.get('baseMapTheme', '')
            vp = self.qgisCaps.visibilityPresets.get(th)
            if not vp:
                gws.log.warning(f'map theme {th!r} not found')
                return []
            return vp

        if bt == 'singleLayer':
            uid = self.caps.globalProps.get('baseMapLayer', '')
            if not uid:
                return []
            return [uid]

        return []

    def layer_entry(self, sl: gws.SourceLayer) -> t.Optional[LayerEntry]:
        qf_props = {}

        for k, v in sl.properties.items():
            if k.startswith('QFieldSync/'):
                qf_props[k.split('/').pop()] = v

        # 'offline', 'no_action' or 'remove'
        qf_action = qf_props.get('action', 'no_action')

        prov = sl.dataSource.get('provider')

        if prov == 'postgres':
            if qf_action == 'remove':
                return LayerEntry(action=LayerAction.remove)

            if qf_action == 'offline':
                return self.postgres_layer_entry(sl, qf_props)

        else:
            if sl.sourceId in self.caps.baseMapLayerIds:
                return LayerEntry(action=LayerAction.baseMap)

            if qf_action == 'remove':
                return LayerEntry(action=LayerAction.remove)

            if qf_action == 'offline':
                gws.log.warning(f'layer {sl.sourceId!r}: offline editing of {prov!r} not supported')
                return LayerEntry(action=LayerAction.remove)

    def postgres_layer_entry(self, sl, qf_props):
        read_only = qf_props.get('is_geometry_locked')

        table_name = sl.dataSource.get('table')
        if not table_name or table_name.startswith('(') or table_name.upper().startswith('SELECT '):
            gws.log.warning(f'layer {sl.sourceId!r}: no table name')
            return LayerEntry(action=LayerAction.remove)

        me = self.model_entry_for_table(sl)
        if not me:
            gws.log.warning(f'layer {sl.sourceId!r}: no model')
            return LayerEntry(action=LayerAction.remove)

        if not read_only and not me.model.isEditable:
            gws.log.warning(f'layer {sl.sourceId!r}: table {table_name!r} is not editable')
            return LayerEntry(action=LayerAction.remove)

        return LayerEntry(
            action=LayerAction.edit,
            readOnly=read_only,
            modelEntry=me,
            sqlFilter=sl.dataSource.get('sql', ''),
        )

    def model_entry_for_table(self, sl: gws.SourceLayer) -> t.Optional[ModelEntry]:
        table_name = sl.dataSource.get('table')

        for model in self.package.models:
            full_name = model.provider.join_table_name('', model.tableName)
            if full_name == model.provider.join_table_name('', table_name):
                gp_name = self.gp_name_for_model(full_name)
                if gp_name not in self.caps.modelMap:
                    self.caps.modelMap[gp_name] = self.model_entry(gp_name, model)
                return self.caps.modelMap[gp_name]

        pg_provider = self.package.qgisProvider.postgres_provider_from_datasource(sl.dataSource)
        if not pg_provider.has_table(table_name):
            gws.log.warning(f'layer {sl.sourceId!r}: table {table_name!r} not found')
            return

        gp_name = self.gp_name_for_model(table_name)

        if gp_name not in self.caps.modelMap:
            model = self.package.root.create_shared(
                gws.ext.object.model,
                gws.Config(
                    uid=f'qfield_model_{table_name}',
                    type='postgres',
                    # NB: permissions are checked in the public export/import functions above
                    permissions=gws.Config(read=gws.PUBLIC, edit=gws.PUBLIC),
                    tableName=table_name,
                    isEditable=True,
                    _defaultProvider=pg_provider,
                )
            )
            self.caps.modelMap[gp_name] = self.model_entry(gp_name, model)

        return self.caps.modelMap[gp_name]

    def model_entry(self, gp_name: str, model: gws.IDatabaseModel) -> ModelEntry:
        return ModelEntry(
            gpId=len(self.caps.modelMap),
            gpName=gp_name,
            model=model,
            fidToPkey={},
            columnIndex={},
            editOperations=[],
            features=[],
        )

    def gp_name_for_model(self, table_name):
        if '.' not in table_name:
            table_name = 'public.' + table_name
        return 'qm_' + table_name.replace('.', '_')


##


def _write_offline_log(ol: OfflineLog, conn: sa.Connection):
    """Create logging tables for the Qgis Offline editing feature.

    see qgis/src/core/qgsofflineediting.cpp createLoggingTables()
    """

    meta = sa.MetaData()
    tables = _offline_log_tables(meta)

    meta.create_all(conn)

    for name, tab in tables.items():
        data = getattr(ol, name, [])
        if data:
            conn.execute(sa.insert(tab), data)


def _read_offline_log(conn: sa.Connection) -> OfflineLog:
    meta = sa.MetaData()
    tables = _offline_log_tables(meta)

    ol = OfflineLog()

    for name, tab in tables.items():
        sel = sa.select(tab)
        setattr(ol, name, [r._asdict() for r in conn.execute(sel)])

    by_commit = lambda r: r['commit_no']

    ol.log_added_attrs.sort(key=by_commit)
    ol.log_geometry_updates.sort(key=by_commit)
    ol.log_geometry_updates.sort(key=by_commit)

    return ol


def _offline_log_tables(meta: sa.MetaData) -> dict[str, sa.Table]:
    ddl = '''
        log_added_attrs (layer_id INT, commit_no INT, name TEXT, type INT, length INT, precision INT, comment TEXT) 
        log_added_features (layer_id INT, fid INT) 
        log_feature_updates (layer_id INT, commit_no INT, fid INT, attr INT, value TEXT) 
        log_fids (layer_id INT, offline_fid INT, remote_fid INT, remote_pk TEXT) 
        log_geometry_updates (layer_id INT, commit_no INT, fid INT, geom_wkt TEXT) 
        log_indices (name TEXT, last_index INT) 
        log_layer_ids (id INT, qgis_id TEXT) 
        log_removed_features (layer_id INT, fid INT) 
    '''

    # our extensions

    ddl += 'log_gws_columns (layer_id INT, attr INT, name TEXT)\n'
    ddl += 'log_gws_tables (layer_id INT, name TEXT)\n'

    tables = {}

    for ln in ddl.strip().split('\n'):
        name, rest = ln.split(maxsplit=1)
        args = [name, meta]
        for f in rest.strip()[1:-1].split(','):
            fname, ftype = f.split()
            fn = None
            if ftype == 'INT':
                fn = sa.Integer
            if ftype == 'TEXT':
                fn = sa.String
            args.append(sa.Column(fname, fn))
        tables[name] = sa.Table(*args)

    return tables
