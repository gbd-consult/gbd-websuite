"""Raster image layer."""

from typing import Optional

import fnmatch

import gws
import gws.base.shape
import gws.base.layer
import gws.lib.image
import gws.lib.osx
import gws.lib.bounds
import gws.lib.crs
import gws.gis.zoom
import gws.gis.ms
import gws.gis.gdalx

gws.ext.new.layer('picture')


class ProviderConfig(gws.Config):
    """Raster data provider."""

    type: str
    """Type"""
    paths: Optional[list[gws.FilePath]]
    """List of image file paths."""
    pathPattern: Optional[str]
    """Glob pattern for image file paths."""
    crs: Optional[gws.CrsName]
    """Default CRS for the images."""


class Config(gws.base.layer.Config):
    """Raster layer"""

    provider: ProviderConfig
    """Raster provider"""
    processing: Optional[list[str]]
    """Processing options (https://mapserver.org/input/raster.html#special-processing-directives)"""


class ImageEntry(gws.Data):
    path: str
    bounds: gws.Bounds


class Provider(gws.Data):
    paths: list[str]
    dirname: str
    pattern: str


class Object(gws.base.layer.image.Object):
    tileIndexPath: str
    entries: list[ImageEntry]
    processing: list[str]

    def configure(self):
        self.configure_layer()
        self.processing = self.cfg('processing', default=[])

    def configure_provider(self):
        p = self.cfg('provider')

        paths = []
        if p.paths:
            paths = p.paths
        elif p.pathPattern:
            v = gws.lib.osx.parse_path(p.pathPattern)
            paths = sorted(gws.lib.osx.find_files(v['dirname'], fnmatch.translate(v['filename'])))

        default_crs = gws.lib.crs.require(p.crs) if p.crs else self.parentBounds.crs

        self.entries = self._enum_images(paths, default_crs)
        if not self.entries:
            raise gws.ConfigurationError('no images found')

        self.tileIndexPath = self._make_tile_index()

    def _enum_images(self, paths, default_crs):
        es1 = []

        for path in paths:
            try:
                with gws.gis.gdalx.open_raster(path, default_crs=default_crs) as gd:
                    es1.append(ImageEntry(path=path, bounds=gd.bounds()))
            except gws.gis.gdalx.Error as exc:
                gws.log.warning(f'image: {path!r}: ERROR: ({exc})')

        if not es1:
            return []

        # all images must have the same CRS
        es2 = []
        crs = es1[0].bounds.crs
        for e in es1:
            if e.bounds.crs == crs:
                es2.append(e)
                continue
            gws.log.warning(f'image: {e.path!r}: ERROR: wrong crs {e.bounds.crs}, must be {crs}')

        return es2

    def _make_tile_index(self):
        idx_name = f'tile_index_{self.uid}'
        idx_path = f'{gws.c.OBJECT_CACHE_DIR}/{idx_name}.shp'

        records = []

        for e in self.entries:
            records.append(
                gws.FeatureRecord(
                    attributes={'location': e.path},
                    shape=gws.base.shape.from_bounds(e.bounds),
                )
            )

        with gws.gis.gdalx.open_vector(idx_path, 'w') as ds:
            la = ds.create_layer(
                name=idx_name,
                columns={'location': gws.AttributeType.str},
                geometry_type=gws.GeometryType.polygon,
                crs=self.mapCrs,
            )
            la.insert(records)
            ds.gdDataset.ExecuteSQL(f'CREATE SPATIAL INDEX ON {idx_name}')

        gws.log.debug(f'tile_index: layer {self.uid} created {idx_path=}')
        return idx_path

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        b = gws.lib.bounds.union([e.bounds for e in self.entries])
        self.bounds = gws.lib.bounds.transform(b, self.parentBounds.crs)
        return True

    def configure_grid(self):
        p = self.cfg('grid', default=gws.Config())

        self.grid = gws.TileGrid(
            origin=p.origin or gws.Origin.nw,
            tileSize=p.tileSize or 256,
            bounds=self.bounds,
        )

        if p.resolutions:
            self.grid.resolutions = p.resolutions
        else:
            self.grid.resolutions = gws.gis.zoom.resolutions_from_bounds(self.grid.bounds, self.grid.tileSize)

    ##

    # @TODO check memory usage
    MAX_BOX_SIZE = 9000

    def render(self, lri):
        ts = gws.u.mstime()

        ms_map = gws.gis.ms.new_map()

        ms_map.add_raster_layer(
            gws.gis.ms.RasterLayerOptions(
                tileIndex=self.tileIndexPath,
                crs=self.entries[0].bounds.crs,
                processing=self.processing,
            )
        )

        if lri.type == gws.LayerRenderInputType.box:

            def get_box(bounds, width, height):
                img = ms_map.draw(bounds, (width, height))
                if self.root.app.developer_option('mapserver.save_temp_maps'):
                    gws.u.write_debug_file(f'ms_{self.uid}_{gws.u.microtime()}.map', ms_map.to_string())
                return img.to_bytes()

            content = gws.base.layer.util.generic_render_box(self, lri, get_box, box_size=self.MAX_BOX_SIZE)
            return gws.LayerRenderOutput(content=content)

        if lri.type == gws.LayerRenderInputType.xyz:
            ext = self.bounds.extent
            w = (ext[2] - ext[0]) / (1 << lri.z)

            x0 = ext[0] + lri.x * w
            x1 = x0 + w

            y0 = ext[3] - (lri.y + 1) * w
            y1 = y0 + w

            img = ms_map.draw(
                gws.lib.bounds.from_extent((x0, y0, x1, y1), crs=self.bounds.crs),
                (self.grid.tileSize, self.grid.tileSize),
            )
            if self.root.app.developer_option('mapserver.save_temp_maps'):
                gws.u.write_debug_file(f'ms_{self.uid}_{gws.u.microtime()}.map', ms_map.to_string())

            if self.root.app.developer_option('map.annotate_render'):
                ts = gws.u.mstime() - ts
                text = f'{lri.z} : {lri.x} / {lri.y}\nUID={self.uid}\n{ts}ms'
                img.add_text(text, x=5, y=5).add_box()

            content = img.to_bytes(self.imageFormat.mimeTypes[0], self.imageFormat.options)

            return gws.LayerRenderOutput(content=content)
