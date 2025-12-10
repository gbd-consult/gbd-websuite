from typing import cast
import shutil

import gws
import gws.plugin.qgis.project
import gws.gis.gdalx
import gws.lib.jsonx
import gws.lib.image
import gws.lib.bounds
import gws.lib.osx as osx
import gws.lib.extent
import gws.gis.render
import gws.gis.zoom

from . import core, caps as caps_mod


class Args(gws.Data):
    uid: str
    qfcProject: core.QfcProject
    caps: caps_mod.Caps
    project: gws.Project
    user: gws.User
    packageDir: str
    mapCacheDir: str
    withBaseMap: bool
    withData: bool
    withMedia: bool
    withQgis: bool


class Object:
    uid: str
    root: gws.Root
    qfcProject: core.QfcProject
    project: gws.Project
    user: gws.User
    args: Args
    caps: caps_mod.Caps

    def create_package(self, root: gws.Root, args: Args):
        self.root = root
        self.uid = args.uid
        self.pathMap = {}

        self.args = args
        self.qfcProject = self.args.qfcProject
        self.project = self.args.project
        self.user = self.args.user
        self.caps = args.caps

        if self.args.withData:
            self.write_data()

        if self.args.withBaseMap:
            self.write_base_map()

        if self.args.withMedia:
            self.write_media()

        if self.args.withQgis:
            self.write_qgis_project()

        gws.lib.jsonx.to_path(
            f'{self.args.packageDir}/path_map.json',
            self.pathMap,
            pretty=True,
        )

    def write_data(self):
        for le in self.caps.layerMap.values():
            if le.action != caps_mod.LayerAction.edit:
                continue
            if le.dataSourceFileName in self.pathMap:
                continue
            path = f'{self.args.packageDir}/{le.dataSourceFileName}'
            self.pathMap[le.dataSourceFileName] = path
            with gws.gis.gdalx.open_vector(path, 'w') as ds:
                self.write_features(le, ds)

    def write_base_map(self):
        # @TODO options for flattened base maps
        for le in self.caps.layerMap.values():
            if le.action == caps_mod.LayerAction.baseMap:
                self.write_base_map_layer(le)

    def write_media(self):
        for d in self.caps.copyDirs:
            if not gws.u.is_dir(d):
                gws.log.warning(f'{self.uid}: media dir not found: {d!r}')
                continue
            if self.caps.qgisPath:
                rel_dir = osx.rel_path(d, self.caps.qgisPath)
            else:
                # @TODO absolute dir with a postgres-based qgis project?
                rel_dir = d.split('/')[-1]
            for p in osx.find_files(d):
                self.pathMap[rel_dir + '/' + osx.rel_path(p, d)] = p

    #

    def get_features_for_layer(self, le: caps_mod.LayerEntry) -> list[gws.Feature]:
        me = le.modelEntry
        q = gws.SearchQuery()
        if self.caps.copyOnlyAreaOfInterest:
            q.bounds = gws.u.require(self.caps.areaOfInterest or self.qfcProject.qgisProvider.bounds)
        mc = gws.ModelContext(user=self.user, project=self.project, op=gws.ModelOperation.read)
        return me.model.find_features(q, mc)

    _SUPPORTED_ATTRIBUTE_TYPES = {
        gws.AttributeType.bool,
        gws.AttributeType.date,
        gws.AttributeType.datetime,
        gws.AttributeType.float,
        gws.AttributeType.int,
        gws.AttributeType.str,
        gws.AttributeType.time,
    }

    def write_features(self, le: caps_mod.LayerEntry, ds: gws.gis.gdalx.VectorDataSet):
        features = self.get_features_for_layer(le)

        me = le.modelEntry
        gws.log.debug(f'{self.uid}: {self.qfcProject.uid}::{me.gpName!r} BEGIN write_features')

        columns = {}
        for f in me.model.fields:
            if f.attributeType not in self._SUPPORTED_ATTRIBUTE_TYPES:
                continue
            if f.name.lower() == 'fid':
                gws.log.warning(f'{self.uid}: {self.qfcProject.uid}::{me.gpName!r} skipping "fid" {f.name=}')
                continue
            columns[f.name] = f.attributeType

        gp_layer = ds.create_layer(
            me.gpName,
            columns=columns,
            geometry_type=me.model.geometryType,
            crs=me.model.geometryCrs,
            overwrite=True,
        )
        
        records = [
            gws.FeatureRecord(
                attributes={name: feature.get(name) for name in columns},
                shape=feature.shape(),
                meta={},
            )
            for feature in features
        ]

        with ds.transaction():
            gp_layer.insert(records)

        gws.log.debug(f'{self.uid}: {self.qfcProject.uid}::{me.gpName!r} END write_features, count={gp_layer.count()}')

    ##

    def write_base_map_layer(self, le: caps_mod.LayerEntry):
        max_zoom = max(
            self.caps.projectProps.baseMapTilesMinZoomLevel or 0,
            self.caps.projectProps.baseMapTilesMaxZoomLevel or 0,
        )
        if max_zoom < 3:
            gws.log.warning(f'{self.uid}: write_base_map_layer: invalid zoom level {max_zoom=}')
            max_zoom = 3
        if max_zoom > 20:
            gws.log.warning(f'{self.uid}: write_base_map_layer: invalid zoom level {max_zoom=}')
            max_zoom = 20
        
        cache_path = f'{self.args.mapCacheDir}/{max_zoom}_{le.dataSourceFileName}'
        age = osx.file_age(cache_path)
        ttl = self.qfcProject.mapCacheLifeTime
        gws.log.debug(f'{self.uid}: write_base_map_layer: {le.qgisId}: {cache_path=} {ttl=}/{age=}')

        if ttl > 0 and (0 < age < ttl):
            gws.log.debug(f'{self.uid}: write_base_map_layer: CACHED!')
            self.pathMap[le.dataSourceFileName] = cache_path
            return

        bounds = gws.u.require(self.caps.areaOfInterest or self.qfcProject.qgisProvider.bounds)
        bounds = gws.lib.bounds.transform(bounds, le.sourceLayer.supportedCrs[0])

        ls = list(reversed(gws.gis.zoom.OSM_RESOLUTIONS))
        resolution = ls[max_zoom]

        w, h = gws.lib.extent.size(bounds.extent)
        px_size = (w / resolution, h / resolution, gws.Uom.px)

        gws.log.debug(f'{self.uid}: write_base_map_layer: {max_zoom=} {resolution=} {px_size=} {bounds=}')

        flat_layer = cast(
            gws.Layer,
            self.qfcProject.root.create_temporary(
                gws.ext.object.layer,
                type='qgisflat',
                _parentBounds=bounds,
                _parentResolutions=[1],
                _defaultProvider=self.qfcProject.qgisProvider,
                _defaultSourceLayers=[le.sourceLayer],
            ),
        )

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

        lro = gws.u.require(flat_layer.render(lri))
        img = gws.lib.image.from_bytes(lro.content)

        with gws.gis.gdalx.open_from_image(img, bounds) as src:
            src.create_copy(cache_path)

        self.pathMap[le.dataSourceFileName] = cache_path

    def write_qgis_project(self):
        root_el = self.qfcProject.qgisProvider.qgis_project().xml_root()
        QgisXmlTransformer().run(self, root_el)
        xml = root_el.to_string()
        xml = self.replace_vars(xml)
        fname = f'{self.qfcProject.uid}.qgs'
        path = f'{self.args.packageDir}/{fname}'
        gws.u.write_file(path, xml)
        self.pathMap[fname] = path

    def replace_vars(self, s: str) -> str:
        # @TODO render attributes as templates
        s = s.replace('{user.authToken}', self.user.authToken)
        s = s.replace('{user.loginName}', self.user.loginName)
        s = s.replace('{user.displayName}', self.user.displayName)
        return s


class QgisXmlTransformer:
    po: Object
    root: gws.XmlElement
    toRemove: list[gws.XmlElement]

    def run(self, po: Object, root_el: gws.XmlElement):
        self.po = po
        self.root = root_el
        self.toRemove = []

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
        # p = properties.add('OfflineEditingPlugin').add('OfflineDbPath', type='QString')
        # p.text = f'{self.po.deviceDbPath}'

        # ensure relative paths
        p = properties.find('Paths/Absolute')
        if not p:
            p = properties.add('Paths').add('Absolute', type='bool')
        p.text = 'false'

    def update_layer_tree(self):
        for el in self.root.findall('.//layer-tree-layer'):
            le = self.po.caps.layerMap.get(el.get('id'))
            if not le:
                continue

            if le.action == caps_mod.LayerAction.remove:
                self.toRemove.append(el)
                continue

            el.set('source', le.dataSource)
            el.set('providerKey', le.dataProvider)

    def update_map_layers(self):
        for el in self.root.findall('.//maplayer'):
            le = self.po.caps.layerMap.get(el.textof('id'))
            if not le:
                continue

            if le.action == caps_mod.LayerAction.remove:
                self.toRemove.append(el)
                continue

            el.require('datasource').text = le.dataSource
            el.require('provider').text = le.dataProvider

            if le.action == caps_mod.LayerAction.edit:
                cp = el.find('customproperties')
                if cp:
                    el.remove(cp)
                opt = el.add('customproperties').add('Option', type='Map')

                opt.add('Option', type='QString', name='QFieldSync/action', value='offline')
                opt.add('Option', type='QString', name='QFieldSync/attachment_naming', value='{}')
                opt.add('Option', type='QString', name='QFieldSync/photo_naming', value='{}')
                opt.add('Option', type='QString', name='QFieldSync/sourceDataPrimaryKeys', value='fid')

                if le.action == caps_mod.LayerAction.edit:
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

            le = self.po.caps.layerMap.get(ref_id)
            if not le or le.action != caps_mod.LayerAction.edit:
                gws.log.warning(f'{self.po.uid}: relation: referenced layer not found: {ref_id!r}')
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

            le = self.po.caps.layerMap.get(ref_id)
            if not le or le.action != caps_mod.LayerAction.edit:
                gws.log.warning(f'{self.po.uid}: editWidget: referenced layer not found: {ref_id!r}')
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
            if sub.tag == 'layer-tree-layer' and sub not in self.toRemove:
                is_empty = False

        if is_empty:
            self.toRemove.append(group_el)

        return not is_empty

    def remove_elements(self, el, parent_el):
        if el in self.toRemove:
            parent_el.remove(el)
            return
        ns = el.children()
        for n in ns:
            self.remove_elements(n, el)
