"""Creates QField packages from QGIS projects."""

from typing import cast, Optional

import gws
import gws.lib.gdalx
import gws.lib.jsonx
import gws.lib.image
import gws.lib.bounds
import gws.lib.osx as osx
import gws.lib.extent
import gws.gis.render
import gws.lib.grid

from . import core, caps as caps_mod

PATH_MAP_FILE = 'path_map.json'
"""Name of the file that maps package file names to paths on disk."""
COMPLETE_FILE = 'package_complete'
"""Name of the marker file written when a package is complete."""


class Args(gws.Data):
    """Arguments for the packager."""

    uid: str
    """Package uid, used in log messages."""
    qfcProject: core.QfcProject
    """QField project."""
    caps: caps_mod.Caps
    """Capabilities of the QField project."""
    project: Optional[gws.Project]
    """GWS project context."""
    user: gws.User
    """User the package is created for."""
    packageDir: str
    """Directory to write the package into."""
    mapCacheDir: str
    """Directory for cached base map images."""
    withBaseMap: bool
    """Render base maps."""
    withData: bool
    """Write the data of offline layers."""
    withMedia: bool
    """Add media files."""
    withQgis: bool
    """Write the modified QGIS project."""


class Object:
    """Packager, writes a QField package into a directory.

    Data files and the QGIS project are written into the package directory.
    Base maps and media files are not copied; the package path map refers to
    them where they are. The path map is written into ``PATH_MAP_FILE``, and
    ``COMPLETE_FILE`` marks the package as complete.
    """

    uid: str
    """Package uid."""
    root: gws.Root
    """Root object."""
    qfcProject: core.QfcProject
    """QField project."""
    project: Optional[gws.Project]
    """GWS project context."""
    user: gws.User
    """User the package is created for."""
    args: Args
    """Packager arguments."""
    caps: caps_mod.Caps
    """Capabilities of the QField project."""

    def create_package(self, root: gws.Root, args: Args):
        """Create a package.

        Args:
            root: Root object.
            args: Packager arguments.
        """
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
            f'{self.args.packageDir}/{PATH_MAP_FILE}',
            self.pathMap,
            pretty=True,
        )
        gws.u.write_file(
            f'{self.args.packageDir}/{COMPLETE_FILE}',
            '1',
        )

    def write_data(self):
        """Write the features of each ``edit`` layer into a GeoPackage file."""
        for le in self.caps.layerMap.values():
            if le.action != caps_mod.LayerAction.edit:
                continue
            if le.dataSourceFileName in self.pathMap:
                continue
            path = f'{self.args.packageDir}/{le.dataSourceFileName}'
            self.pathMap[le.dataSourceFileName] = path
            with gws.lib.gdalx.open_vector(path, 'w') as ds:
                self.write_features(le, ds)

    def write_base_map(self):
        """Render all ``baseMap`` layers."""
        # @TODO options for flattened base maps
        for le in self.caps.layerMap.values():
            if le.action == caps_mod.LayerAction.baseMap:
                self.write_base_map_layer(le)

    def write_media(self):
        """Add the files from the directories to copy to the path map.

        Paths in the package are relative to the QGIS project file. If the project is not stored in a file, the last directory name is used.
        """
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
        """Read the features of an ``edit`` layer.

        If only the area of interest is copied, features are limited to the area of interest, or the project bounds.

        Args:
            le: Layer entry.

        Returns:
            Features.
        """
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

    def write_features(self, le: caps_mod.LayerEntry, ds: gws.lib.gdalx.VectorDataSet):
        """Write the features of a layer into a GeoPackage layer.

        Only fields of simple types are written. A field named ``fid`` is written as ``fid_gws``, because GDAL uses ``fid`` internally.

        Args:
            le: Layer entry.
            ds: GeoPackage data set.
        """
        features = self.get_features_for_layer(le)

        me = le.modelEntry
        gws.log.debug(f'{self.uid}: {self.qfcProject.uid}::{me.gpName!r} BEGIN write_features')

        columns = {}
        field_names = {}
        for f in me.model.fields:
            if f.attributeType not in self._SUPPORTED_ATTRIBUTE_TYPES:
                continue
            # FID field needs to be renamed, because GDAL uses it internally
            name = f.name
            if f.name.lower() == 'fid':
                name = 'fid_gws'
            columns[name] = f.attributeType
            field_names[name] = f.name

        gp_layer = ds.create_layer(
            me.gpName,
            columns=columns,
            geometry_type=me.model.geometryType,
            crs=me.model.geometryCrs,
            overwrite=True,
        )

        records = []
        for feat in features:
            rec = gws.FeatureRecord(
                attributes={name: feat.get(field_name) for name, field_name in field_names.items()},
                shape=feat.shape(),
                meta={},
            )
            records.append(rec)

        with ds.transaction():
            gp_layer.insert(records)

        gws.log.debug(f'{self.uid}: {self.qfcProject.uid}::{me.gpName!r} END write_features, count={gp_layer.count()}')

    ##

    def write_base_map_layer(self, le: caps_mod.LayerEntry):
        """Render a base map layer into a raster file.

        The layer is rendered as a single image covering the area of interest
        (or the project bounds), at the resolution of the maximum zoom level
        (clamped to 3..20) in the grid of the target CRS. The file is stored in the
        map cache directory and reused while it is younger than ``mapCacheLifeTime``.

        Args:
            le: Layer entry.
        """
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
        bounds = gws.lib.bounds.transform(bounds, self.qfcProject.qgisProvider.forceCrs)

        resolution = gws.lib.grid.resolution_for_level(gws.lib.grid.for_crs(bounds.crs), max_zoom)

        w, h = gws.lib.extent.size(bounds.extent)
        px_size = (w / resolution, h / resolution, gws.Uom.px)

        gws.log.debug(f'{self.uid}: write_base_map_layer: {max_zoom=} {resolution=} {px_size=} {bounds=}')

        flat_layer = cast(
            gws.Layer,
            self.qfcProject.root.create_temporary(
                gws.ext.object.layer,
                type='qgisflat',
                _parentWgsExtent=gws.lib.bounds.wgs_extent(bounds),
                _mapCrs=bounds.crs,
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
            targetCrs=bounds.crs,
            user=self.user,
            view=mv,
        )

        lro = gws.u.require(flat_layer.render(lri))
        img = gws.lib.image.from_bytes(lro.content).convert('RGBA')

        with gws.lib.gdalx.open_from_image(img, bounds) as src:
            src.save_as(cache_path)

        self.pathMap[le.dataSourceFileName] = cache_path

    def write_qgis_project(self):
        """Write the modified QGIS project into the package.

        The original project is also written next to it, with the extension ``.source.qgs``.
        """
        fname = f'{self.qfcProject.uid}.qgs'
        path = f'{self.args.packageDir}/{fname}'

        qp = self.qfcProject.qgisProvider.qgis_project()
        gws.u.write_file(path + '.source.qgs', qp.text)

        root_el = qp.xml_root()
        QgisXmlTransformer().run(self, root_el)
        xml = root_el.to_string()
        xml = self.replace_vars(xml)
        gws.u.write_file(path, xml)
        self.pathMap[fname] = path

    def replace_vars(self, s: str) -> str:
        """Replace user variables in a string.

        Supported variables are ``{user.authToken}``, ``{user.loginName}`` and ``{user.displayName}``.

        Args:
            s: Source string.

        Returns:
            String with the variables replaced.
        """
        # @TODO render attributes as templates
        s = s.replace('{user.authToken}', self.user.authToken)
        s = s.replace('{user.loginName}', self.user.loginName)
        s = s.replace('{user.displayName}', self.user.displayName)
        return s


class QgisXmlTransformer:
    """Modifies the QGIS project XML for the package.

    Points layers and references to the packaged data sources, removes layers
    with the ``remove`` action and empty layer groups, and makes paths relative.
    """

    po: Object
    """Packager."""
    root: gws.XmlElement
    """Root element of the project."""
    toRemove: list[gws.XmlElement]
    """Elements to remove."""

    def run(self, po: Object, root_el: gws.XmlElement):
        """Transform the project XML in place.

        Args:
            po: Packager.
            root_el: Root element of the project.
        """
        self.po = po
        self.root = root_el
        self.toRemove = []

        self.change_global_props()
        self.update_layer_tree()
        self.update_map_layers()
        self.update_referenced_layers()
        self.update_edit_widgets()

        self.cleanup_layer_group(root_el.find('layer-tree-group'))
        self.remove_elements(root_el, None)

    def change_global_props(self):
        """Set the project to use relative paths."""
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
        """Update the data source of layer tree entries, or mark them for removal."""
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
        """Update the data source of map layers, or mark them for removal."""
        for el in self.root.findall('.//maplayer'):
            le = self.po.caps.layerMap.get(el.textof('id'))
            if not le:
                continue

            if le.action == caps_mod.LayerAction.remove:
                self.toRemove.append(el)
                continue

            el.require('datasource').text = le.dataSource
            el.require('provider').text = le.dataProvider

            # @TODO do we need to change properties at all?
            #
            # if le.action == caps_mod.LayerAction.edit:
            #     cp = el.find('customproperties')
            #     if cp:
            #         el.remove(cp)
            #     opt = el.add('customproperties').add('Option', type='Map')

            #     opt.add('Option', type='QString', name='QFieldSync/action', value='offline')
            #     opt.add('Option', type='QString', name='QFieldSync/attachment_naming', value='{}')
            #     opt.add('Option', type='QString', name='QFieldSync/photo_naming', value='{}')
            #     opt.add('Option', type='QString', name='QFieldSync/sourceDataPrimaryKeys', value='fid')

            #     if le.action == caps_mod.LayerAction.edit:
            #         if le.readOnly:
            #             opt.add('Option', type='bool', name='QFieldSync/is_geometry_locked', value='true')
            #         else:
            #             opt.add('Option', type='bool', name='isOfflineEditable', value='true')

    def update_referenced_layers(self):
        """Update the data source in relations that reference ``edit`` layers."""
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
        """Update the data source in relation reference widgets that reference ``edit`` layers."""
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
        """Mark empty layer groups for removal, recursively.

        Args:
            group_el: Layer tree group element.

        Returns:
            ``True`` if the group contains layers that are kept.
        """
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
        """Remove the elements marked for removal, recursively.

        Args:
            el: Element to check.
            parent_el: Parent element.
        """
        if el in self.toRemove:
            parent_el.remove(el)
            return
        ns = el.children()
        for n in ns:
            self.remove_elements(n, el)
