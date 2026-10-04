"""Image set provider for raster layers."""

import fnmatch
from typing import Optional

import gws
import gws.base.shape
import gws.lib.osx
import gws.lib.crs
import gws.lib.gdalx


class Config(gws.Config):
    """Set of georeferenced image files for a raster layer."""

    paths: Optional[list[gws.FilePath]]
    """Image files to show."""
    pathPattern: Optional[str]
    """Glob pattern for image files."""
    crs: Optional[gws.CrsName]
    """CRS for images that have none."""


class ImageEntry(gws.Data):
    """A georeferenced image file."""

    path: str
    """File path."""
    bounds: gws.Bounds
    """Image bounds in the image CRS."""


class Object(gws.Node):
    """Set of georeferenced image files, given as paths or a glob pattern."""

    paths: list[str]
    """Image file paths."""
    crs: Optional[gws.Crs]
    """CRS for images that have none."""

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
            pp = gws.lib.osx.parse_path(p)
            self.paths = sorted(
                gws.lib.osx.find_files(
                    pp.dirname,
                    fnmatch.translate(pp.filename),
                )
            )
            return

        raise gws.ConfigurationError('no paths or pathPattern specified for raster provider.')

    def cache_hash(self):
        """Compute a hash of the provider settings.

        Returns:
            Hash string, built from the paths and the CRS.
        """
        return gws.u.sha256([
            self.paths,
            self.crs.srid if self.crs else '',
        ])

    def enumerate_images(self, default_crs: gws.Crs) -> list[ImageEntry]:
        """Read the bounds of the image files.

        Files that cannot be opened, and files in a CRS other than the CRS of
        the first image, are skipped with a configuration warning.

        Args:
            default_crs: CRS for images that have none.

        Returns:
            Image entries, all in the same CRS.
        """
        es1 = []

        for path in self.paths:
            try:
                with gws.lib.gdalx.open_raster(path, default_crs=default_crs) as gd:
                    es1.append(ImageEntry(path=path, bounds=gd.bounds()))
            except gws.lib.gdalx.Error as exc:
                self.root.config_warning(f'raster_provider: {path!r}: cannot open: ({exc})')

        if not es1:
            return []

        # all images must have the same CRS
        es2 = []
        crs = es1[0].bounds.crs
        for e in es1:
            if e.bounds.crs == crs:
                es2.append(e)
                continue
            self.root.config_warning(f'raster_provider: {e.path!r}: wrong crs {e.bounds.crs}, must be {crs}')

        return es2

    def make_tile_index(self, entries: list[ImageEntry], file_name: str) -> str:
        """Create a MapServer tile index shapefile for the images.

        The index has a polygon per image with the file path in the
        ``location`` column, and a spatial index.

        Args:
            entries: Images to index.
            file_name: Base name of the shapefile, in the object cache directory.

        Returns:
            Path to the shapefile.
        """
        idx_path = f'{gws.c.OBJECT_CACHE_DIR}/{file_name}.shp'

        records = []

        for e in entries:
            records.append(
                gws.FeatureRecord(
                    attributes={'location': e.path},
                    shape=gws.base.shape.from_bounds(e.bounds),
                )
            )

        with gws.lib.gdalx.open_vector(idx_path, 'w') as ds:
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
