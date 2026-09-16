"""Base raster grabber."""

import os
from typing import Optional

import gws
import gws.lib.extent
import gws.lib.gdalx
import gws.lib.grid
import gws.lib.crs
import gws.lib.image
import gws.lib.mime
import gws.gis.cache

_BlobImagePair = tuple[bytes | None, gws.Image | None]

MAX_LEVEL = 30
"""Serving stopper: levels beyond this are never served or precomputed."""

EPHEMERAL_MAX_AGE = 60
"""Lifetime (seconds) of tiles in the ephemeral store."""

BLOCK_LOCK_TIMEOUT = 60
"""Seconds to wait for a block being composed by another process."""


class Options(gws.Data):
    crs: gws.Crs
    cache: gws.MapCache
    extent: Optional[gws.Extent]
    imageFormat: gws.ImageFormat
    provider: Optional[gws.ServiceProvider]


class Object(gws.Grabber):
    """Base raster grabber."""

    grid: gws.MapGrid
    rangeForLevel: dict[int, gws.MapTileRange]
    mime: str
    store: gws.gis.cache.store.Object
    """Persistent store, for levels within the cache settings."""
    ephemeralStore: gws.gis.cache.store.Object
    """Short-lived store, for everything else, so that blocks are composed once."""
    cache: gws.MapCache
    extent: gws.Extent
    """Extent in the target CRS."""
    minLevel: int
    """Coarsest served level."""
    maxLevel: int
    """Finest served level."""
    requestTiles: int
    """Tiles per side composed in one block; 1 means no meta-tiling."""

    targetCrs: gws.Crs
    """Target crs, defines the grabber CRS."""
    imageFormat: gws.ImageFormat
    """Format tiles are stored and returned in."""
    sourceCrs: gws.Crs
    """Source crs, defines the original CRS of the raster data."""

    def __init__(self, opts: Options):
        self.targetCrs = opts.crs
        self.grid = gws.lib.grid.for_crs(self.targetCrs)

        self.imageFormat = opts.imageFormat
        self.mime = self.imageFormat.mimeTypes[0]

        self.cache = opts.cache
        self.requestTiles = 1
        self.store = gws.gis.cache.store.Object(
            f'{gws.c.MAP_CACHE_DIR}/{self.cache.name}',
            max_age=self.cache.maxAge,
            extension=gws.lib.mime.extension_for(self.mime),
        )
        self.ephemeralStore = gws.gis.cache.store.Object(
            gws.u.ephemeral_dir(f'tiles_{self.cache.name}'),
            max_age=EPHEMERAL_MAX_AGE,
            extension=gws.lib.mime.extension_for(self.mime),
        )

        self.extent = opts.extent or gws.lib.extent.transform_from_wgs(self.targetCrs.wgsMaxExtent, self.targetCrs)
        self.minLevel = 0
        self.maxLevel = MAX_LEVEL

        self.rangeForLevel = {}
        for z in range(self.minLevel, self.maxLevel + 1):
            rng = gws.lib.grid.range_for_extent(self.grid, self.extent, z)
            if not rng:
                raise gws.ConfigurationError(f'grabber {self.cache.name!r}: empty tile range for level {z}')
            self.rangeForLevel[z] = rng

    ##

    def levels(self):
        return list(self.rangeForLevel)

    def tile_range_for_level(self, z):
        return self.rangeForLevel[z]

    def get_tile_as_bytes(self, tile, params=None):
        return self.as_bytes(self.get_tile_as_pair(tile, params))

    def get_tile_as_image(self, tile, params=None):
        return self.as_image(self.get_tile_as_pair(tile, params))

    def get_tiles_as_bytes_dict(self, tr, params=None):
        return {mt: self.get_tile_as_bytes(mt, params) for mt in self.valid_tiles_in_range(tr)}

    def get_tiles_as_image_dict(self, tr, params=None):
        return {mt: self.get_tile_as_image(mt, params) for mt in self.valid_tiles_in_range(tr)}

    def get_box_as_bytes(self, extent, width, height, params=None):
        return self.as_bytes(self.get_box_as_pair(extent, width, height, params))

    def get_box_as_image(self, extent, width, height, params=None):
        return self.as_image(self.get_box_as_pair(extent, width, height, params))

    ##

    def get_tile_as_pair(self, tile: gws.MapTile, params: dict | None) -> _BlobImagePair:
        z = tile[-1]
        if not self.is_serving(z):
            return self.empty_tile(), None

        if not gws.lib.grid.in_range(tile, self.tile_range_for_level(z)):
            return self.empty_tile(), None

        if params:
            block_images = self.compose_tile_block_as_image_dict(tile, params)
            return None, block_images[tile]

        blob = self.store_read(tile)
        if blob is not None:
            return blob, None

        try:
            bx, by, _ = self.block_start_tile(tile)
            with gws.u.server_lock(f'grabber_{self.cache.name}_{z}_{bx}_{by}', BLOCK_LOCK_TIMEOUT):
                return self.get_block_and_return_pair(tile)
        except gws.LockBusyError:
            gws.log.warning(f'grabber {self.cache.name!r}: block lock busy for {tile!r}')
            return self.empty_tile(), None

    def get_block_and_return_pair(self, tile: gws.MapTile) -> _BlobImagePair:
        out = self.empty_tile(), None
        block_images = self.compose_tile_block_as_image_dict(tile)
        for bt, img in block_images.items():
            blob = self.as_bytes((None, img))
            self.store_write(bt, blob)
            if bt == tile:
                out = blob, img
        return out

    def get_box_as_pair(self, extent: gws.Extent, width, height, params: dict | None) -> tuple[bytes | None, gws.Image | None]:
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        z = gws.lib.grid.level_for_resolution(self.grid, (extent[2] - extent[0]) / w)
        if params or not self.is_storing(z):
            return None, self.compose_box_as_image(extent, w, h, params)

        rng = gws.lib.grid.range_for_extent(self.grid, extent, z)
        if not rng:
            return self.empty_box(w, h), None

        x0, y0, x1, y1, _ = rng
        ts = self.grid.tileSize
        mosaic = gws.lib.image.from_size(((x1 - x0 + 1) * ts, (y1 - y0 + 1) * ts))
        for (tx, ty, _), img in self.get_tiles_as_image_dict(rng).items():
            mosaic.paste(img, ((tx - x0) * ts, (ty - y0) * ts))

        mosaic_extent = gws.lib.grid.extent_for_range(self.grid, rng)
        with gws.lib.gdalx.open_from_image(mosaic, gws.Bounds(crs=self.targetCrs, extent=mosaic_extent)) as ds:
            return None, ds.warp_to_image(
                dict(
                    dstSRS=self.targetCrs.epsg,
                    outputBounds=extent,
                    outputBoundsSRS=self.targetCrs.epsg,
                    width=w,
                    height=h,
                    resampleAlg='bilinear',
                )
            )

    ##

    def compose_tile_as_image(self, tile: gws.MapTile, params: dict | None = None) -> gws.Image:
        return self.compose_box_as_image(
            gws.lib.grid.extent_for_tile(self.grid, tile),
            self.grid.tileSize,
            self.grid.tileSize,
            params,
        )

    def compose_tile_block_as_image_dict(self, tile: gws.MapTile, params: dict | None = None) -> dict[gws.MapTile, gws.Image]:
        if self.requestTiles != 1:
            raise NotImplementedError(f'compose_tile_block_as_image_dict not implemented in {self!r}')
        return {tile: self.compose_tile_as_image(tile, params)}

    def compose_box_as_image(self, extent: gws.Extent, width: int, height: int, params: dict | None = None) -> gws.Image:
        raise NotImplementedError(f'compose_box_as_image not implemented in {self!r}')

    ##

    def as_bytes(self, bi: _BlobImagePair) -> bytes:
        blob, img = bi
        if blob is not None:
            return blob
        if img is not None:
            return img.to_bytes(self.mime, self.imageFormat.options)
        raise gws.Error('unexpected state')

    def as_image(self, bi: _BlobImagePair) -> gws.Image:
        blob, img = bi
        if img is not None:
            return img
        if blob is not None:
            return gws.lib.image.from_bytes(blob)
        raise gws.Error('unexpected state')

    def normalize_image(self, img: gws.Image, width: int, height: int) -> gws.Image:
        """Ensure a source image is RGBA and has the requested size."""

        if img.size() != (width, height):
            raise gws.ExternalServiceError(f'grabber {self.cache.name!r}: unexpected image size {img.size()!r}, expected {(width, height)!r}')
        return img.convert('RGBA')

    def valid_tiles_in_range(self, tr):
        z = tr[-1]
        if not self.is_serving(z):
            return []
        level_rng = self.tile_range_for_level(z)
        return [mt for mt in gws.lib.grid.enum_tiles(tr) if gws.lib.grid.in_range(mt, level_rng)]

    def block_start_tile(self, mt: gws.MapTile) -> gws.MapTile:
        x, y, z = mt
        n = self.requestTiles
        return (x // n) * n, (y // n) * n, z

    def extent_to_source_crs(self, extent: gws.Extent) -> gws.Extent | None:
        wgs_extent = gws.lib.extent.transform_to_wgs(extent, self.targetCrs)
        wgs_extent = self.sourceCrs.clip_extent(wgs_extent)
        if not wgs_extent:
            return
        src_extent = gws.lib.extent.transform_from_wgs(wgs_extent, self.sourceCrs)
        if not gws.lib.extent.is_valid(src_extent):
            return
        return src_extent

    def warp_image(self, img: gws.Image, src_extent: gws.Extent, target_extent: gws.Extent, w: int, h: int) -> gws.Image:
        with gws.lib.gdalx.open_from_image(img, gws.Bounds(crs=self.sourceCrs, extent=src_extent)) as ds:
            return ds.warp_to_image(
                dict(
                    dstSRS=self.targetCrs.epsg,
                    outputBounds=target_extent,
                    outputBoundsSRS=self.targetCrs.epsg,
                    width=w,
                    height=h,
                    resampleAlg='bilinear',
                    warpOptions=['XSCALE=1', 'YSCALE=1'],
                )
            )

    def is_serving(self, z):
        return self.minLevel <= z <= self.maxLevel

    def is_storing(self, z):
        return self.cache.maxAge > 0 and z <= self.cache.maxLevel

    def store_read(self, mt: gws.MapTile):
        st = self.store if self.is_storing(mt[-1]) else self.ephemeralStore
        return st.read(mt)

    def store_write(self, mt: gws.MapTile, blob: bytes):
        st = self.store if self.is_storing(mt[-1]) else self.ephemeralStore
        st.write(mt, blob)
        gws.u.ephemeral_cleanup()

    def empty_tile(self) -> bytes:
        if not hasattr(self, '_emptyTile'):
            self._emptyTile = self.empty_box(self.grid.tileSize, self.grid.tileSize)
        return self._emptyTile

    def empty_box(self, width, height) -> bytes:
        return self.as_bytes((None, gws.lib.image.from_size((width, height))))
