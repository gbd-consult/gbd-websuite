"""Raster image layer."""

from typing import Optional

import fnmatch

import gws
import gws.base.shape
import gws.base.layer
import gws.lib.osx
import gws.lib.bounds
import gws.lib.crs
import gws.gis.zoom
import gws.gis.ms
import gws.gis.ms.util
import gws.gis.gdalx

gws.ext.new.layer('raster')


class ProviderConfig(gws.Config):
    """Raster data provider."""

    type: str
    """Type ('file')."""
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
    """Processing directives."""
    transparentColor: Optional[str]
    """Color to treat as transparent in the layer."""


class ImageEntry(gws.Data):
    path: str
    bounds: gws.Bounds


class Provider(gws.Data):
    paths: list[str]
    dirname: str
    pattern: str


class Object(gws.base.layer.image.Object):
    entries: list[ImageEntry]
    msOptions: gws.gis.ms.LayerOptions

    def configure(self):
        self.msOptions = gws.gis.ms.LayerOptions(
            type=gws.gis.ms.LayerType.raster,
            processing=self.cfg('processing', default=[]),
            transparentColor=self.cfg('transparentColor', default=None),
        )
        self.configure_layer()

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

        self.msOptions.crs = self.entries[0].bounds.crs
        self.msOptions.tileIndex = self._make_tile_index()

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
        self.msOptions.crs = self.bounds.crs
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

    def render(self, lri):
        return gws.gis.ms.util.raster_render(self, lri)
