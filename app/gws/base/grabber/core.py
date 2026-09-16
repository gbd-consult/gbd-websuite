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


DEFAULT_IMAGE_FORMAT = gws.lib.image.FormatConfig(name='png8', mimeTypes=['image/png'], options={'mode': 'P'})

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
    imageFormat: Optional[gws.ImageFormat]
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

    def __init__(self, opts: Options):
        self.targetCrs = opts.crs
        self.grid = gws.lib.grid.for_crs(self.targetCrs)

        p = DEFAULT_IMAGE_FORMAT
        self.imageFormat = opts.imageFormat or gws.ImageFormat(name=p.name, mimeTypes=p.mimeTypes, options=p.options or {})
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
        r = self._get_tile(tile, params)
        if isinstance(r, bytes):
            return r
        return r.to_bytes(self.mime, self.imageFormat.options)

    def get_tile_as_image(self, tile, params=None):
        r = self._get_tile(tile, params)
        if isinstance(r, bytes):
            return gws.lib.image.from_bytes(r)
        return r

    def get_tiles_as_bytes(self, tr, params=None):
        return {mt: self.get_tile_as_bytes(mt, params) for mt in self.tiles_in_range(tr)}

    def get_tiles_as_images(self, tr, params=None):
        return {mt: self.get_tile_as_image(mt, params) for mt in self.tiles_in_range(tr)}

    def get_box_as_bytes(self, extent, width, height, params=None):
        r = self._get_box(extent, width, height, params)
        if isinstance(r, bytes):
            return r
        return r.to_bytes(self.mime, self.imageFormat.options)

    def get_box_as_image(self, extent, width, height, params=None):
        r = self._get_box(extent, width, height, params)
        if isinstance(r, bytes):
            return gws.lib.image.from_bytes(r)
        return r

    def tiles_in_range(self, tr):
        z = tr[-1]
        if not self.is_serving(z):
            return []
        rng = self.tile_range_for_level(z)
        return [mt for mt in gws.lib.grid.enum_tiles(tr) if gws.lib.grid.in_range(mt, rng)]

    ##

    def _get_tile(self, tile: gws.MapTile, params: dict | None) -> bytes | gws.Image:
        z = tile[-1]
        if not self.is_serving(z):
            return self.empty_tile()

        if not gws.lib.grid.in_range(tile, self.tile_range_for_level(z)):
            return self.empty_tile()

        blob = self.store_read(tile, params)
        if blob is not None:
            return blob

        if params:
            return self.compose_block_as_images(tile, params)[tile]

        x, y, z = tile
        n = self.requestTiles
        bx, by = (x // n) * n, (y // n) * n
        try:
            with gws.u.server_lock(f'grabber_{self.cache.name}_{z}_{bx}_{by}', BLOCK_LOCK_TIMEOUT):
                blob = self.store_read(tile, params)
                if blob is not None:
                    return blob
                return self._compose_and_store_block(tile, params)
        except gws.LockBusyError:
            return self._compose_and_store_block(tile, params)

    def _compose_and_store_block(self, tile: gws.MapTile, params: dict | None) -> gws.Image:
        images = self.compose_block_as_images(tile, params)
        for mt, img in images.items():
            self.store_write_image(mt, img, params)
        return images[tile]

    def _get_box(self, extent: gws.Extent, width, height, params: dict | None) -> bytes | gws.Image:
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        z = gws.lib.grid.level_for_resolution(self.grid, (extent[2] - extent[0]) / w)
        if params or not self.is_storing(z):
            return self.compose_box_as_image(extent, w, h, params)

        rng = gws.lib.grid.range_for_extent(self.grid, extent, z)
        if not rng:
            return self.empty_box(w, h)

        x0, y0, x1, y1, _ = rng
        ts = self.grid.tileSize
        mosaic = gws.lib.image.from_size(((x1 - x0 + 1) * ts, (y1 - y0 + 1) * ts))
        for (tx, ty, _), img in self.get_tiles_as_images(rng).items():
            mosaic.paste(img, ((tx - x0) * ts, (ty - y0) * ts))

        mosaic_extent = gws.lib.grid.extent_for_range(self.grid, rng)
        with gws.lib.gdalx.open_from_image(mosaic, gws.Bounds(crs=self.targetCrs, extent=mosaic_extent)) as ds:
            return ds.warp_to_image(
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

    def compose_block_as_images(self, tile: gws.MapTile, params: dict | None = None) -> dict[gws.MapTile, gws.Image]:
        img = self.compose_box_as_image(
            gws.lib.grid.extent_for_tile(self.grid, tile),
            self.grid.tileSize,
            self.grid.tileSize,
            params,
        )
        return {tile: img}

    def compose_box_as_bytes(self, extent: gws.Extent, width: int, height: int, params: dict | None = None) -> bytes:
        img = self.compose_box_as_image(extent, width, height, params)
        return img.to_bytes(self.mime, self.imageFormat.options)

    def compose_box_as_image(self, extent: gws.Extent, width: int, height: int, params: dict | None = None) -> gws.Image:
        raise NotImplementedError(f'compose_box_as_image not implemented in {self!r}')

    ##

    def is_serving(self, z):
        return self.minLevel <= z <= self.maxLevel

    def is_storing(self, z):
        return self.cache.maxAge > 0 and z <= self.cache.maxLevel

    def store_for(self, z: int, params: dict | None = None) -> gws.TileStore | None:
        if params:
            return None
        if self.is_storing(z):
            return self.store
        return self.ephemeralStore

    def store_read(self, mt: gws.MapTile, params: dict | None = None):
        st = self.store_for(mt[2], params)
        if st is None:
            return None
        return st.read(mt)

    def store_write(self, mt: gws.MapTile, blob: bytes, params: dict | None = None):
        st = self.store_for(mt[2], params)
        if st is None:
            return
        st.write(mt, blob)
        if st is self.ephemeralStore:
            gws.u.ephemeral_cleanup()

    def store_write_image(self, mt: gws.MapTile, img: gws.Image, params: dict | None = None):
        self.store_write(mt, img.to_bytes(self.mime, self.imageFormat.options), params)

    def empty_tile(self) -> bytes:
        if not hasattr(self, '_emptyTile'):
            self._emptyTile = self.empty_box(self.grid.tileSize, self.grid.tileSize)
        return self._emptyTile

    def empty_box(self, width, height) -> bytes:
        img = gws.lib.image.from_size((width, height))
        return img.to_bytes(self.mime, self.imageFormat.options)
