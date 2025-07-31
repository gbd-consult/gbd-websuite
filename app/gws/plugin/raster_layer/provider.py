"""Raster image provider."""

import fnmatch
from typing import Optional

import gws
import gws.base.shape
import gws.lib.osx
import gws.lib.crs
import gws.gis.gdalx


class Config(gws.Config):
    """Raster data provider."""

    paths: Optional[list[gws.FilePath]]
    """List of image file paths."""
    pathPattern: Optional[str]
    """Glob pattern for image file paths."""
    crs: Optional[gws.CrsName]
    """Default CRS for the images."""


class ImageEntry(gws.Data):
    path: str
    bounds: gws.Bounds


class Object(gws.Node):
    paths: list[str]
    crs: Optional[gws.Crs]

    def configure(self):
        p = self.cfg('crs')
        self.crs = gws.lib.crs.require(p) if p else None

        self.paths = []

        p = self.cfg('paths')
        if p:
            self.paths = p
            return

        p = self.cfg('pathPattern')
        if p:
            v = gws.lib.osx.parse_path(p)
            self.paths = sorted(gws.lib.osx.find_files(v['dirname'], fnmatch.translate(v['filename'])))
            return

        raise gws.ConfigurationError('no paths or pathPattern specified for raster provider.')

    def enumerate_images(self, default_crs: gws.Crs) -> list[ImageEntry]:
        es1 = []

        for path in self.paths:
            try:
                with gws.gis.gdalx.open_raster(path, default_crs=default_crs) as gd:
                    es1.append(ImageEntry(path=path, bounds=gd.bounds()))
            except gws.gis.gdalx.Error as exc:
                gws.log.warning(f'raster_provider: {path!r}: ERROR: ({exc})')

        if not es1:
            return []

        # all images must have the same CRS
        es2 = []
        crs = es1[0].bounds.crs
        for e in es1:
            if e.bounds.crs == crs:
                es2.append(e)
                continue
            gws.log.warning(f'raster_provider: {e.path!r}: ERROR: wrong crs {e.bounds.crs}, must be {crs}')

        return es2

    def make_tile_index(self, entries: list[ImageEntry], file_name: str) -> str:
        idx_path = f'{gws.c.OBJECT_CACHE_DIR}/{file_name}.shp'

        records = []

        for e in entries:
            records.append(
                gws.FeatureRecord(
                    attributes={'location': e.path},
                    shape=gws.base.shape.from_bounds(e.bounds),
                )
            )

        with gws.gis.gdalx.open_vector(idx_path, 'w') as ds:
            la = ds.create_layer(
                name=file_name,
                columns={'location': gws.AttributeType.str},
                geometry_type=gws.GeometryType.polygon,
                crs=entries[0].bounds.crs,
            )
            la.insert(records)
            ds.gdDataset.ExecuteSQL(f'CREATE SPATIAL INDEX ON {file_name}')

        gws.log.debug(f'raster_provider: created {idx_path=}')
        return idx_path
