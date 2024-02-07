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


##

class ExportOptions(gws.Data):
    baseDir: str
    qgisFileName: str
    dbFileName: str
    withBaseMap: bool
    withData: bool
    withMedia: bool
    withQgis: bool
    storeDbInDCIM: bool


class ImportOptions(gws.Data):
    baseDir: str
    dbFileName: str


def export_package(package: Package, project: gws.IProject, user: gws.IUser, opts: ExportOptions):
    """Write QField package files to a directory."""

    gws.log.debug(f'qfield: export: {package=} {project=} {opts=}')

    if not user.can_read(package):
        raise gws.ForbiddenError(f'cannot read {package.uid=}')

    _Exporter().run(package, project, user, opts)


def import_data_from_package(package: Package, project: gws.IProject, user: gws.IUser, opts: ImportOptions):
    """Read QField package files from a directory and update the database."""

    gws.log.debug(f'qfield: import: {package=} {project=} {opts=}')

    if not user.can_write(package):
        raise gws.ForbiddenError(f'cannot write to {package.uid=}')

    _Importer().run(package, project, user, opts)


##

class _LayerAction(t.Enum):
    remove = 'remove'
    edit = 'edit'
    baseMap = 'baseMap'


class _EditAction(gws.Enum):
    update = 'update'
    geometryUpdate = 'geometryUpdate'
    insert = 'insert'
    delete = 'delete'


class _EditOperation(gws.Data):
    action: _EditAction
    fid: int
    pkey: str
    columnName: str
    attributes: dict


class _LayerEntry(gws.Data):
    action: _LayerAction
    gpId: int
    qgisId: str
    model: gws.IDatabaseModel
    fidToPkey: dict
    columnIndex: dict
    editOperations: list[_EditOperation]
    localPath: str
    devicePath: str
    readOnly: bool
    dataSource: str
    dataProvider: str
    sourceLayer: gws.SourceLayer


class _QfCaps(gws.Data):
    qgisPath: str
    layerMap: dict[str, _LayerEntry]
    globalProps: dict
    dirsToCopy: list[str]
    baseMapLayerIds: list[str]


class _OfflineLog(gws.Data):
    log_added_attrs: list[dict]
    log_added_features: list[dict]
    log_feature_updates: list[dict]
    log_fids: list[dict]
    log_geometry_updates: list[dict]
    log_indices: list[dict]
    log_layer_ids: list[dict]
    log_removed_features: list[dict]
    log_gws_columns: list[dict]


_GP_ATTRIBUTE_TYPES = {
    gws.AttributeType.bool,
    gws.AttributeType.date,
    gws.AttributeType.datetime,
    gws.AttributeType.float,
    gws.AttributeType.int,
    gws.AttributeType.str,
    gws.AttributeType.time,
}


class _Exporter:
    package: Package
    project: gws.IProject
    user: gws.IUser
    opts: ExportOptions

    sourceQgisProject: gws.plugin.qgis.project.Object

    targetQgisPath: str

    deviceDbPath: str
    localDbPath: str

    qfCaps: _QfCaps

    def run(self, package: Package, project: gws.IProject, user: gws.IUser, opts: ExportOptions):
        self.package = package
        self.project = project
        self.user = user
        self.opts = opts

        self.sourceQgisProject = self.package.qgisProvider.qgis_project()

        self.targetQgisPath = f'{self.opts.baseDir}/{self.opts.qgisFileName or self.package.uid}.qgs'

        db_path = f'{self.opts.dbFileName or self.package.uid}.{GPKG_EXT}'
        if self.opts.storeDbInDCIM:
            db_path = f'DCIM/{db_path}'
            gws.ensure_dir(f'{self.opts.baseDir}/DCIM')

        self.localDbPath = f'{self.opts.baseDir}/{db_path}'
        self.deviceDbPath = f'./{db_path}'

        self.qfCaps = _QfCapsParser().run(self.package)

        for le in self.qfCaps.layerMap.values():
            if le.action == _LayerAction.edit:
                le.localPath = self.localDbPath
                le.devicePath = self.deviceDbPath
                le.dataSource = f'{le.devicePath}|layername={le.qgisId}'
                le.dataProvider = f'ogr'
            if le.action == _LayerAction.baseMap:
                le.localPath = f'{self.opts.baseDir}/{le.qgisId}.{GPKG_EXT}'
                le.devicePath = f'./{le.qgisId}.{GPKG_EXT}'
                le.dataSource = f'{le.devicePath}'
                le.dataProvider = f'gdal'

        if self.opts.withData:
            with gws.gis.gdalx.open(self.localDbPath, 'w', as_vector=True) as ds:
                for le in self.qfCaps.layerMap.values():
                    if le.action == _LayerAction.edit:
                        self.write_features(le, ds)
            self.write_offline_log()

        if self.opts.withBaseMap:
            # @TODO options for flattened base maps
            for le in self.qfCaps.layerMap.values():
                if le.action == _LayerAction.baseMap:
                    self.write_base_map(le)

        if self.opts.withMedia:
            for d in self.qfCaps.dirsToCopy:
                if self.qfCaps.qgisPath:
                    rel_dir = gws.lib.osx.rel_path(d, self.qfCaps.qgisPath)
                else:
                    # @TODO absolute dir with a postgres-based qgis project?
                    rel_dir = d.split('/')[-1]
                shutil.copytree(d, f'{self.opts.baseDir}/{rel_dir}')

        if self.opts.withQgis:
            self.write_qgis_project()

    def write_features(self, le: _LayerEntry, ds: gws.gis.gdalx.DataSet):
        # see qgis/src/core/qgsofflineediting.cpp convertToOfflineLayer()

        gp_fields = [fld for fld in le.model.fields if fld.attributeType in _GP_ATTRIBUTE_TYPES]

        gp_layer = ds.create_layer(
            le.qgisId,
            columns={fld.name: fld.attributeType for fld in gp_fields},
            geometry_type=le.model.geometryType,
            crs=le.model.geometryCrs,
            overwrite=True
        )

        mc = gws.ModelContext(user=self.user, project=self.project, op=gws.ModelOperation.read)
        features = le.model.find_features(gws.SearchQuery(), mc)
        records = []

        for feature in features:
            props = le.model.feature_to_props(feature, mc)
            records.append(gws.FeatureRecord(
                attributes={fld.name: props.attributes.get(fld.name) for fld in gp_fields},
                shape=feature.shape(),
                meta={}
            ))

        with ds.transaction():
            fids = gp_layer.insert(records)

        le.columnIndex = {}
        col_idx = 1  # column 0 == fid
        for fld in gp_fields:
            le.columnIndex[col_idx] = fld.name
            col_idx += 1

        le.fidToPkey = {}
        if le.model.uidName:
            for rec, fid in zip(records, fids):
                le.fidToPkey[fid] = rec.attributes.get(le.model.uidName)

        gws.log.debug(f'{self.opts.baseDir}: write_features: {self.package.uid}::{le.qgisId!r} count={gp_layer.count()}')

    def write_base_map(self, le: _LayerEntry):
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
        gws.gis.gdalx.create_copy(le.localPath, src)

    def write_offline_log(self):
        editable_layers = [
            le
            for le in self.qfCaps.layerMap.values()
            if le.action == _LayerAction.edit
        ]

        ol = _OfflineLog()

        ol.log_fids = [
            {'layer_id': le.gpId, 'offline_fid': fid, 'remote_fid': fid, 'remote_pk': str(pk)}
            for le in editable_layers
            for fid, pk in le.fidToPkey.items()
        ]
        ol.log_indices = [
            {'name': 'commit_no', 'last_index': 0},
            {'name': 'layer_id', 'last_index': len(editable_layers)}
        ]
        ol.log_layer_ids = [
            {'id': le.gpId, 'qgis_id': le.qgisId}
            for le in editable_layers
        ]
        ol.log_gws_columns = [
            {'layer_id': le.gpId, 'attr': col_id, 'name': col_name}
            for le in editable_layers
            for col_id, col_name in le.columnIndex.items()
        ]

        engine = sa.create_engine(f'sqlite:///{self.localDbPath}', echo=False, future=True)
        with engine.begin() as conn:
            _write_offline_log(ol, conn)

    def write_qgis_project(self):
        root_el = self.sourceQgisProject.xml_root()
        _QqsXmlTransformer().run(self, root_el)
        gws.write_file(self.targetQgisPath, root_el.to_string())


class _QqsXmlTransformer:
    ex: _Exporter
    root: gws.IXmlElement
    remove: list[gws.IXmlElement]

    def run(self, ex: _Exporter, root_el: gws.IXmlElement):
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

            if le.action == _LayerAction.remove:
                self.remove.append(el)
                continue

            el.set('source', le.dataSource)
            el.set('providerKey', le.dataProvider)

    def update_map_layers(self):
        for el in self.root.findall('.//maplayer'):
            le = self.ex.qfCaps.layerMap.get(el.textof('id'))
            if not le:
                continue

            if le.action == _LayerAction.remove:
                self.remove.append(el)
                continue

            el.find('datasource').text = le.dataSource
            el.find('provider').text = le.dataProvider

            if le.action == _LayerAction.edit:
                el.remove(el.find('customproperties'))
                opt = el.add('customproperties').add('Option', type='Map')

                opt.add('Option', type='QString', name='QFieldSync/action', value='offline')
                opt.add('Option', type='QString', name='QFieldSync/attachment_naming', value='{}')
                opt.add('Option', type='QString', name='QFieldSync/photo_naming', value='{}')
                opt.add('Option', type='QString', name='QFieldSync/sourceDataPrimaryKeys', value='fid')

                if le.action == _LayerAction.edit:
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
                        layerId="LAYER_ID"
                        referencedLayer="LAYER_ID" 
                        providerKey="<REPLACE THIS>"
                        dataSource="<REPLACE THIS>"
                    >
                    ...
            """

            qid = el.get('referencedLayer')
            le = self.ex.qfCaps.layerMap.get(el.get('id'))
            if not le or le.action != _LayerAction.edit:
                gws.log.warning(f'layer not found {qid!r} in relation')
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
                    <Option value="<LAYER ID>" name="ReferencedLayerId" type="QString"/>
                    <Option value="<REPLACE THIS>" name="ReferencedLayerDataSource" type="QString"/>
                    <Option value="<REPLACE THIS>" name="ReferencedLayerProviderKey" type="QString"/>
                    ...
            """

            if el.get('type') != 'RelationReference':
                continue

            qid = None
            for opt in el.findall('.//Option'):
                if opt.get('name') == 'ReferencedLayerId':
                    qid = opt.get('value')
                    break

            if not qid:
                continue

            le = self.ex.qfCaps.layerMap.get(el.get('id'))
            if not le or le.action != _LayerAction.edit:
                gws.log.warning(f'layer not found {qid!r} in editWidget')
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

class _Importer:
    package: Package
    project: gws.IProject
    user: gws.IUser
    opts: ImportOptions

    localDbPath: str
    localImagePaths: list[str]

    qfCaps: _QfCaps

    def run(self, package: Package, project: gws.IProject, user: gws.IUser, opts: ImportOptions):
        self.package = package
        self.project = project
        self.user = user
        self.opts = opts

        self.localDbPath = ''
        self.localImagePaths = []

        db_name = f'{self.opts.dbFileName or self.package.uid}.{GPKG_EXT}'

        for path in gws.lib.osx.find_files(self.opts.baseDir):
            if path.endswith(db_name):
                self.localDbPath = path
            if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.localImagePaths.append(path)

        gws.log.debug(f'{self.localDbPath=} {self.localImagePaths=}')
        self.qfCaps = _QfCapsParser().run(self.package)

        if self.localDbPath:
            self.import_features()

    def import_features(self):
        updated_layers = self.read_updated_layers()
        for le in updated_layers:
            self.commit_edits(le)

    def read_updated_layers(self):
        engine = sa.create_engine(f'sqlite:///{self.localDbPath}', echo=False, future=True)
        updated_layers = []

        with engine.begin() as conn:
            ol = _read_offline_log(conn)

        for rec in ol.log_layer_ids:
            qgis_id = rec.get('qgis_id')

            le = self.qfCaps.layerMap.get(qgis_id)
            if not le:
                gws.log.warning(f'offline layer {qgis_id!r}: not found')
                continue
            if le.action != _LayerAction.edit:
                gws.log.warning(f'offline layer {qgis_id!r}: not editable')
                continue

            le.gpId = rec.get('id')
            has_updates = self.read_offline_log_for_layer(le, ol)
            if has_updates:
                updated_layers.append(le)

        with gws.gis.gdalx.open(self.localDbPath, 'r', as_vector=True) as ds:
            for le in updated_layers:
                self.read_data_for_layer(le, ds)

        for le in updated_layers:
            self.read_images_for_layer(le)

        return updated_layers

    def read_offline_log_for_layer(self, le: _LayerEntry, ol: _OfflineLog) -> bool:

        le.columnIndex = {}
        for rec in ol.log_gws_columns:
            if rec['layer_id'] == le.gpId:
                le.columnIndex[rec.get('attr')] = rec.get('name')

        le.fidToPkey = {}
        for rec in ol.log_fids:
            if rec['layer_id'] == le.gpId:
                le.fidToPkey[rec.get('offline_fid')] = rec.get('remote_pk')

        le.editOperations = []
        self.read_edit_operations_for_layer(le, ol)

        return len(le.editOperations) > 0

    def read_edit_operations_for_layer(self, le: _LayerEntry, ol: _OfflineLog):
        # NB the order is updates (sorted by commit_no) -> inserts -> deletes

        ops = {}
        for rec in ol.log_feature_updates:
            if rec['layer_id'] == le.gpId:
                fid = rec.get('fid')
                pkey = le.fidToPkey.get(fid)
                cname = le.columnIndex.get(rec.get('attr'))
                if not pkey or not cname:
                    gws.log.warning(f'invalid update record {rec!r}')
                    continue
                ops[pkey, cname] = _EditOperation(action=_EditAction.update, fid=fid, pkey=pkey, columnName=cname)
        le.editOperations.extend(ops.values())

        ops = {}
        for rec in ol.log_geometry_updates:
            if rec['layer_id'] == le.gpId:
                fid = rec.get('fid')
                pkey = le.fidToPkey.get(fid)
                if not pkey:
                    gws.log.warning(f'invalid update record {rec!r}')
                    continue
                ops[pkey] = _EditOperation(action=_EditAction.geometryUpdate, fid=fid, pkey=pkey)
        le.editOperations.extend(ops.values())

        for rec in ol.log_added_features:
            if rec['layer_id'] == le.gpId:
                fid = rec.get('fid')
                le.editOperations.append(_EditOperation(action=_EditAction.insert, fid=fid))

        for rec in ol.log_removed_features:
            if rec['layer_id'] == le.gpId:
                pkey = le.fidToPkey.get(rec.get('fid'))
                if not pkey:
                    gws.log.warning(f'invalid delete record {rec!r}')
                    continue
                le.editOperations.append(_EditOperation(action=_EditAction.delete, pkey=pkey))

    def read_data_for_layer(self, le: _LayerEntry, ds: gws.gis.gdalx.DataSet):

        gp_layer = ds.layer(le.gpId)

        for eo in le.editOperations:
            eo.attributes = {le.model.uidName: eo.pkey}

            if eo.action == _EditAction.delete:
                continue

            feature_rec = gp_layer.get_one(eo.fid, encoding='utf8')
            if not feature_rec:
                gws.log.warning(f'{self.opts.baseDir}: operation {eo!r}: fid not found')
                continue

            if eo.action == _EditAction.update:
                eo.attributes[eo.columnName] = feature_rec.attributes.get(eo.columnName)
                continue

            if eo.action == _EditAction.geometryUpdate:
                if not le.model.geometryName:
                    gws.log.warning(f'{self.opts.baseDir}: operation {eo!r}: geometry not defined')
                    continue
                if not feature_rec.shape:
                    gws.log.warning(f'{self.opts.baseDir}: operation {eo!r}: geometry not found')
                    continue
                eo.attributes[le.model.geometryName] = feature_rec.shape
                continue

            if eo.action == _EditAction.insert:
                eo.attributes.update(feature_rec.attributes)
                if le.model.geometryName and feature_rec.shape:
                    eo.attributes[le.model.geometryName] = feature_rec.shape
                # remove primary key if it is null
                if eo.attributes.get(le.model.uidName) is None:
                    eo.attributes.pop(le.model.uidName, None)
                continue

    def read_images_for_layer(self, le: _LayerEntry):
        file_field_map = {}

        for fld in le.model.fields:
            if fld.extType == 'file':
                file_field_map[t.cast(gws.plugin.model_field.file.Object, fld).cols.name.name] = fld

        if not file_field_map:
            return

        for eo in le.editOperations:
            eo.attributes = self.update_file_attributes(eo, file_field_map)

    def update_file_attributes(self, eo: _EditOperation, file_field_map):
        atts = {}

        for name, val in eo.attributes.items():
            fld = file_field_map.get(name)
            if not fld:
                atts[name] = val
                continue

            fv = self.file_value_for_path(val)
            if not fv:
                gws.log.warning(f'file not found: {val!r}')
                continue

            atts[fld.name] = fv

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

    def commit_edits(self, le: _LayerEntry):
        for eo in le.editOperations:
            gws.log.debug(f'edit: {self.localDbPath}: {le.qgisId=}: {eo}')

            if eo.action in {_EditAction.update, _EditAction.geometryUpdate}:
                mc = gws.ModelContext(user=self.user, op=gws.ModelOperation.update)
                model = self.check_model(le, self.user, gws.Access.write)
                feature = model.feature_from_props(gws.FeatureProps(attributes=eo.attributes), mc)
                if not model.validate_feature(feature, mc):
                    gws.log.info(f'validation errors: {feature.errors}')
                    raise gws.BadRequestError('Validation error')
                model.update_feature(feature, mc)

            if eo.action == _EditAction.insert:
                mc = gws.ModelContext(user=self.user, op=gws.ModelOperation.create)
                model = self.check_model(le, self.user, gws.Access.create)
                feature = model.feature_from_props(gws.FeatureProps(attributes=eo.attributes), mc)
                if not model.validate_feature(feature, mc):
                    gws.log.info(f'validation errors: {feature.errors}')
                    raise gws.BadRequestError('Validation error')
                model.create_feature(feature, mc)

            if eo.action == _EditAction.delete:
                mc = gws.ModelContext(user=self.user, op=gws.ModelOperation.delete)
                model = self.check_model(le, self.user, gws.Access.delete)
                feature = model.feature_from_props(gws.FeatureProps(attributes=eo.attributes), mc)
                model.delete_feature(feature, mc)

    def check_model(self, le: _LayerEntry, user: gws.IUser, access: gws.Access) -> gws.IModel:
        if not le.model:
            raise gws.ForbiddenError(f'{le.qgisId}: model: not found, {access=} {user=}')
        if not le.model.isEditable:
            raise gws.ForbiddenError(f'{le.qgisId}: model not editable, {access=} {user=}')
        if not user.can(access, le.model):
            raise gws.ForbiddenError(f'{le.qgisId}: model forbidden, {access=} {user=}')
        return le.model


class _QfCapsParser:
    """Read qf-related capabilities from the qgis project."""

    package: Package
    caps: _QfCaps
    qgisCaps: gws.plugin.qgis.caps.Caps

    def run(self, package: Package) -> _QfCaps:
        self.package = package
        self.caps = _QfCaps(layerMap={}, globalProps={})
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
                gws.log.warning(f'base map layer {uid!r} not found')
                return []
            return [uid]

        return []

    def layer_entry(self, sl: gws.SourceLayer) -> t.Optional[_LayerEntry]:
        qf_props = {}

        for k, v in sl.properties.items():
            if k.startswith('QFieldSync/'):
                qf_props[k.split('/').pop()] = v

        # 'offline', 'no_action' or 'remove'
        qf_action = qf_props.get('action', 'no_action')

        prov = sl.dataSource.get('provider')

        if prov == 'postgres':
            if qf_action == 'remove':
                return _LayerEntry(action=_LayerAction.remove)

            if qf_action == 'offline':
                return self.postgres_layer_entry(sl, qf_props)

        else:
            if sl.sourceId in self.caps.baseMapLayerIds:
                return _LayerEntry(action=_LayerAction.baseMap)

            if qf_action == 'remove':
                return _LayerEntry(action=_LayerAction.remove)

            if qf_action == 'offline':
                gws.log.warning(f'layer {sl.sourceId!r}: offline editing of {prov!r} not supported')
                return _LayerEntry(action=_LayerAction.remove)

    def postgres_layer_entry(self, sl, qf_props):
        read_only = qf_props.get('is_geometry_locked')

        table_name = sl.dataSource.get('table')
        if not table_name or table_name.startswith('(') or table_name.upper().startswith('SELECT '):
            gws.log.warning(f'layer {sl.sourceId!r}: no table name')
            return _LayerEntry(action=_LayerAction.remove)

        model = self.model_for_table(sl)

        if not read_only and not model.isEditable:
            gws.log.warning(f'layer {sl.sourceId!r}: table {table_name!r} is not editable')
            return _LayerEntry(action=_LayerAction.remove)

        return _LayerEntry(
            action=_LayerAction.edit,
            readOnly=read_only,
            model=model,
        )

    def model_for_table(self, sl: gws.SourceLayer):
        table_name = sl.dataSource.get('table')

        for model in self.package.models:
            if model.tableName == table_name:
                return model

        pg = self.package.qgisProvider.postgres_provider_from_datasource(sl.dataSource)
        if not pg.has_table(table_name):
            gws.log.warning(f'layer {sl.sourceId!r}: table {table_name!r} not found')
            return _LayerEntry(action=_LayerAction.remove)

        return self.package.root.create_shared(
            gws.ext.object.model,
            gws.Config(
                uid=f'qfield_model_{table_name}',
                type='postgres',
                # NB: permissions are checked in the public export/import functions above
                permissions=gws.Config(read=gws.PUBLIC, edit=gws.PUBLIC),
                tableName=table_name,
                isEditable=True,
                _defaultProvider=pg,
            )
        )


##


def _write_offline_log(ol: _OfflineLog, conn: sa.Connection):
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


def _read_offline_log(conn: sa.Connection) -> _OfflineLog:
    meta = sa.MetaData()
    tables = _offline_log_tables(meta)

    ol = _OfflineLog()

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

    # this is our extension

    ddl += 'log_gws_columns (layer_id INT, attr INT, name TEXT)'

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
