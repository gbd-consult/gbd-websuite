"""Base raster grabber."""

import os

import gws
import gws.lib.grid
import gws.lib.crs
import gws.lib.image
import gws.lib.mime

from . import store

DEFAULT_IMAGE_FORMAT = gws.lib.image.FormatConfig(name='png8', mimeTypes=['image/png'], options={'mode': 'P'})


class Config(gws.Config):
    blockSize: int
    crs: gws.CrsName
    extent: gws.Extent
    imageFormat: gws.lib.image.FormatConfig
    cacheMaxAge: int
    """Maximum age of cached tiles, in seconds."""
    cacheMaxLevel: int
    """Maximum level to cache."""
    cacheBaseDir: str
    """Directory where tiles are stored."""
    cacheUid: str

class Object(gws.Grabber):
    """Base raster grabber."""

    blockSize: int
    grid: gws.MapGrid
    rangeForLevel: dict[int, gws.MapTileRange]
    mime: str
    store: store.Object
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
    cacheMaxAge: int
    """Maximum age of cached tiles, in seconds."""
    cacheMaxLevel: int
    """Maximum level to cache."""
    cacheBaseDir: str
    """Directory where tiles are stored."""
    cacheUid: str
    """Unique cache identifier, used to separate caches of different grabbers."""

    def configure(self):
        p = self.cfg('crs')
        self.targetCrs = gws.lib.crs.require(p) if p else gws.lib.crs.WEBMERCATOR
        self.grid = gws.lib.grid.for_crs(self.targetCrs)

        p = self.cfg('imageFormat') or DEFAULT_IMAGE_FORMAT
        self.imageFormat = gws.ImageFormat(name=p.name, mimeTypes=p.mimeTypes, options=p.options or {})
        self.mime = self.imageFormat.mimeTypes[0]


        self.cacheMaxAge = self.cfg('cacheMaxAge') or 0
        self.cacheMaxLevel = self.cfg('cacheMaxLevel') or 0
        self.cacheUid = self.cfg('cacheUid')
        self.cacheBaseDir = f'{gws.c.CACHE_DIR}/grabber/{self.cacheUid}'
        
        self.blockSize = self.cfg('blockSize') or 1
        self.store = store.Object(self.cacheBaseDir, gws.lib.mime.extension_for(self.mime))

        self.extent = self.cfg('extent') or self.grid.extent
        self.minLevel = 0
        self.maxLevel = 24

        self.rangeForLevel = {}
        for z in range(self.minLevel, self.maxLevel + 1):
            rng = gws.lib.grid.range_for_extent(self.grid, self.extent, z)
            if not rng:
                raise gws.ConfigurationError(f'grabber {self.uid!r}: empty tile range for level {z}')
            self.rangeForLevel[z] = rng

        self._emptyTile = b''

    ##

    # def cache_status(self):
    #     levels = []
    #     for z in range(self.minLevel, self.maxLevel + 1):
    #         d = f'{self.cacheBaseDir}/{z:02d}'
    #         if not os.path.isdir(d):
    #             continue
    #         n = sum(len(fs) for _, _, fs in os.walk(d))
    #         rng = self.rangeForLevel[z]
    #         total = (rng[2] - rng[0] + 1) * (rng[3] - rng[1] + 1) if rng else 0
    #         levels.append(gws.Data(level=z, storedCount=n, totalCount=total))
    #     return gws.Data(uid=self.uid, baseDir=self.cacheBaseDir, levels=levels)

    # def cache_drop(self):
    #     self.store.drop()

    ##

    def is_serving(self, z):
        return self.minLevel <= z <= self.maxLevel

    def is_storing(self, z):
        return self.cacheMaxAge > 0 and z <= self.cacheMaxLevel

    def store_read(self, mt: gws.MapTile):
        if not self.is_storing(mt[2]):
            return None
        return self.store.read(mt, self.cacheMaxAge)

    def store_write(self, mt: gws.MapTile, blob: bytes):
        if not self.is_storing(mt[2]):
            return None
        return self.store.write(mt, blob)

    def empty_tile(self) -> bytes:
        if not self._emptyTile:
            self._emptyTile = self.empty_box(self.grid.tileSize, self.grid.tileSize)
        return self._emptyTile

    def empty_box(self, width, height) -> bytes:
        img = gws.lib.image.from_size((width, height))
        return img.to_bytes(self.mime, self.imageFormat.options)
