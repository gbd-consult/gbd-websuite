"""Base raster grabber."""

import os

import gws
import gws.lib.gdalx
import gws.lib.grid
import gws.lib.crs
import gws.lib.image
import gws.lib.mime
import gws.gis.cache


DEFAULT_IMAGE_FORMAT = gws.lib.image.FormatConfig(name='png8', mimeTypes=['image/png'], options={'mode': 'P'})

MAX_LEVEL = 30
"""Serving stopper: levels beyond this are never served or precomputed."""
DEFAULT_CACHE = gws.LayerCache(name='', maxAge=0, maxLevel=0, requestBuffer=0, requestTiles=0)


class Config(gws.Config):
    crs: gws.CrsName
    extent: gws.Extent
    imageFormat: gws.lib.image.FormatConfig


class Object(gws.Grabber):
    """Base raster grabber."""

    grid: gws.MapGrid
    rangeForLevel: dict[int, gws.MapTileRange]
    mime: str
    store: gws.gis.cache.store.Object
    cache: gws.LayerCache
    extent: gws.Extent
    """Extent in the target CRS."""
    minLevel: int
    """Coarsest served level."""
    maxLevel: int
    """Finest served level."""

    targetCrs: gws.Crs
    """Target crs, defines the grabber CRS."""
    imageFormat: gws.ImageFormat
    """Format tiles are stored and returned in."""

    def configure(self):
        p = self.cfg('crs')
        self.targetCrs = gws.lib.crs.require(p) if p else gws.lib.crs.WEBMERCATOR
        self.grid = gws.lib.grid.for_crs(self.targetCrs)

        p = self.cfg('imageFormat') or DEFAULT_IMAGE_FORMAT
        self.imageFormat = gws.ImageFormat(name=p.name, mimeTypes=p.mimeTypes, options=p.options or {})
        self.mime = self.imageFormat.mimeTypes[0]

        self.cache = self.cfg('_defaultCache') or DEFAULT_CACHE
        self.cache.name = self.cache.name or self.uid
        self.store = gws.gis.cache.store.Object(self.cache, gws.lib.mime.extension_for(self.mime))

        self.extent = self.cfg('extent') or self.grid.extent
        self.minLevel = 0
        self.maxLevel = MAX_LEVEL

        self.rangeForLevel = {}
        for z in range(self.minLevel, self.maxLevel + 1):
            rng = gws.lib.grid.range_for_extent(self.grid, self.extent, z)
            if not rng:
                raise gws.ConfigurationError(f'grabber {self.uid!r}: empty tile range for level {z}')
            self.rangeForLevel[z] = rng

        self._emptyTile = b''

    ##

    def levels(self):
        return list(self.rangeForLevel)

    def tile_range_for_level(self, z):
        return self.rangeForLevel[z]

    def get_tile(self, tile, params=None):
        x, y, z = tile
        if not self.is_serving(z):
            return self.empty_tile()

        if not gws.lib.grid.in_range(tile, self.rangeForLevel[z]):
            return self.empty_tile()

        blob = self.store_read(tile, params)
        if blob is not None:
            return blob

        blob = self.fetch_tile(tile, params)
        self.store_write(tile, blob, params)

        return blob

    def get_tiles(self, tr, params=None):
        z = tr[4]
        if not self.is_serving(z):
            return {}

        rng = self.rangeForLevel[z]
        tiles = {}
        for mt in gws.lib.grid.enum_tiles(tr):
            if gws.lib.grid.in_range(mt, rng):
                tiles[mt] = self.get_tile(mt, params)
        return tiles

    def get_box(self, extent, width, height, params=None):
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        z = gws.lib.grid.level_for_resolution(self.grid, (extent[2] - extent[0]) / w)
        if params or not self.is_storing(z):
            img = self.draw_box(extent, w, h, params)
            return img.to_bytes(self.mime, self.imageFormat.options)

        rng = gws.lib.grid.range_for_extent(self.grid, extent, z)
        if not rng:
            return self.empty_box(w, h)

        x0, y0, x1, y1, _ = rng
        ts = self.grid.tileSize
        mosaic = gws.lib.image.from_size(((x1 - x0 + 1) * ts, (y1 - y0 + 1) * ts))
        for (tx, ty, _), blob in self.get_tiles(rng).items():
            mosaic.paste(gws.lib.image.from_bytes(blob), ((tx - x0) * ts, (ty - y0) * ts))

        mosaic_extent = gws.lib.grid.extent_for_range(self.grid, rng)
        with gws.lib.gdalx.open_from_image(mosaic, gws.Bounds(crs=self.targetCrs, extent=mosaic_extent)) as ds:
            img = ds.warp_to_image(
                dict(
                    dstSRS=self.targetCrs.epsg,
                    outputBounds=extent,
                    outputBoundsSRS=self.targetCrs.epsg,
                    width=w,
                    height=h,
                    resampleAlg='bilinear',
                )
            )

        return img.to_bytes(self.mime, self.imageFormat.options)

    ##

    def fetch_tile(self, tile: gws.MapTile, params: dict | None = None) -> bytes:
        img = self.draw_box(
            gws.lib.grid.extent_for_tile(self.grid, tile),
            self.grid.tileSize,
            self.grid.tileSize,
            params,
        )
        return img.to_bytes(self.mime, self.imageFormat.options)

    def draw_box(self, extent: gws.Extent, width: int, height: int, params: dict | None = None) -> gws.Image:
        raise NotImplementedError(f'draw_box not implemented in {self!r}')

    ##

    def is_serving(self, z):
        return self.minLevel <= z <= self.maxLevel

    def is_storing(self, z):
        return self.cache.maxAge > 0 and z <= self.cache.maxLevel

    def store_read(self, mt: gws.MapTile, params: dict | None = None):
        if params or not self.is_storing(mt[2]):
            return None
        return self.store.read(mt)

    def store_write(self, mt: gws.MapTile, blob: bytes, params: dict | None = None):
        if params or not self.is_storing(mt[2]):
            return None
        return self.store.write(mt, blob)

    def empty_tile(self) -> bytes:
        if not self._emptyTile:
            self._emptyTile = self.empty_box(self.grid.tileSize, self.grid.tileSize)
        return self._emptyTile

    def empty_box(self, width, height) -> bytes:
        img = gws.lib.image.from_size((width, height))
        return img.to_bytes(self.mime, self.imageFormat.options)
