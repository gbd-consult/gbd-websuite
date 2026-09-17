"""Base raster grabber."""

import os

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
    extent: gws.Extent
    imageFormat: gws.ImageFormat


class Object(gws.Grabber):
    """Base raster grabber."""

    defaultEphemeralStore: gws.TileStore
    """Short-lived store for static tiles outside the cache settings, so that blocks are composed once."""
    maxLevel: int
    """Finest served level."""
    mime: str
    """Mime type of ``imageFormat``."""
    minLevel: int
    """Coarsest served level."""
    mtrByLevel: dict[int, gws.MapTileRange]
    """Tile range covered by ``extent``, per served level."""
    requestTiles: int
    """Tiles per side composed in one block; 1 means no meta-tiling."""

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
        self.defaultEphemeralStore = self._ephemeral_store('')

        self.extent = opts.extent
        self.minLevel = 0
        self.maxLevel = MAX_LEVEL

        self.mtrByLevel = {}
        for z in range(self.minLevel, self.maxLevel + 1):
            mtr = gws.lib.grid.range_for_extent(self.grid, self.extent, z)
            if not mtr:
                raise gws.ConfigurationError(f'grabber {self.cache.name!r}: empty tile range for level {z}')
            self.mtrByLevel[z] = mtr

    ##

    def get_tile_as_bytes(self, mt, params=None):
        return self.pair_to_bytes(self._get_tile_as_pair(mt, params))

    def get_tile_as_image(self, mt, params=None):
        return self.pair_to_image(self._get_tile_as_pair(mt, params))

    def get_tiles_as_bytes_dict(self, mtr, params=None):
        return {t: self.get_tile_as_bytes(t, params) for t in self._valid_tiles_in_range(mtr)}

    def get_tiles_as_image_dict(self, mtr, params=None):
        return {t: self.get_tile_as_image(t, params) for t in self._valid_tiles_in_range(mtr)}

    def get_box_as_bytes(self, extent, w, h, params=None):
        return self.pair_to_bytes(self._get_box_as_pair(extent, w, h, params))

    def get_box_as_image(self, extent, w, h, params=None):
        return self.pair_to_image(self._get_box_as_pair(extent, w, h, params))

    def levels(self):
        return list(self.mtrByLevel)

    def tile_range_for_level(self, z):
        return self.mtrByLevel[z]

    ##

    def compose_tile_block_as_image_dict(self, mt: gws.MapTile, params: dict | None = None) -> dict[gws.MapTile, gws.Image]:
        """Compose the block of ``requestTiles`` x ``requestTiles`` tiles containing a tile."""

        if self.requestTiles != 1:
            raise NotImplementedError(f'compose_tile_block_as_image_dict not implemented in {self!r}')
        return {mt: self.compose_tile_as_image(mt, params)}

    def compose_tile_as_image(self, mt: gws.MapTile, params: dict | None = None) -> gws.Image:
        """Compose a single tile over its own extent."""

        return self.compose_box_as_image(
            gws.lib.grid.extent_for_tile(self.grid, mt),
            self.grid.tileSize,
            self.grid.tileSize,
            params,
        )

    def compose_box_as_image(self, extent: gws.Extent, w: int, h: int, params: dict | None = None) -> gws.Image:
        """Compose an image for an arbitrary extent and pixel size from the source."""

        raise NotImplementedError(f'compose_box_as_image not implemented in {self!r}')

    ##

    def pair_to_bytes(self, bi: _BlobImagePair) -> bytes:
        """Return the bytes form of a pair, encoding the image if needed."""

        blob, img = bi
        if blob is not None:
            return blob
        if img is not None:
            return self.to_bytes(img)
        raise gws.Error('unexpected state')

    def pair_to_image(self, bi: _BlobImagePair) -> gws.Image:
        """Return the image form of a pair, decoding the bytes if needed."""

        blob, img = bi
        if img is not None:
            return img
        if blob is not None:
            return self.to_image(blob)
        raise gws.Error('unexpected state')

    def to_bytes(self, img: gws.Image) -> bytes:
        """Encode an image in the grabber's image format."""

        return img.to_bytes(self.mime, self.imageFormat.options)

    def to_image(self, blob: bytes) -> gws.Image:
        """Decode an encoded image."""

        return gws.lib.image.from_bytes(blob)

    def normalize_image(self, img: gws.Image, w: int, h: int) -> gws.Image:
        """Ensure a source image is RGBA and has the requested size."""

        if img.size() != (w, h):
            raise gws.ExternalServiceError(f'grabber {self.cache.name!r}: unexpected image size {img.size()!r}')
        return img.convert('RGBA')

    def block_start_tile(self, mt: gws.MapTile) -> gws.MapTile:
        """Return the first tile of the block containing a tile."""

        x, y, z = mt
        n = self.requestTiles
        return (x // n) * n, (y // n) * n, z

    def extent_to_source_crs(self, extent: gws.Extent) -> gws.Extent | None:
        """Transform a target extent into the source CRS, clipped to the source area of use."""

        wgs_extent = gws.lib.extent.transform_to_wgs(extent, self.targetCrs)
        wgs_extent = self.sourceCrs.clip_wgs_extent(wgs_extent)
        if not wgs_extent:
            return
        src_extent = gws.lib.extent.transform_from_wgs(wgs_extent, self.sourceCrs)
        if not gws.lib.extent.is_valid(src_extent):
            return
        return src_extent

    def warp_image(self, img: gws.Image, src_extent: gws.Extent, target_extent: gws.Extent, w: int, h: int) -> gws.Image:
        """Warp a source image onto a target extent and pixel size."""

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

    def empty_image(self, w: int, h: int) -> gws.Image:
        """Return a transparent image of the given size."""

        return gws.lib.image.from_size((w, h))

    def empty_box(self, w: int, h: int) -> bytes:
        """Return a transparent image of the given size, as encoded bytes."""

        return self.to_bytes(self.empty_image(w, h))

    def empty_tile(self) -> bytes:
        """Return the transparent tile."""

        if not hasattr(self, '_emptyTile'):
            self._emptyTile = self.empty_box(self.grid.tileSize, self.grid.tileSize)
        return self._emptyTile

    def is_serving(self, z: int) -> bool:
        """True if a level is within the serving range."""

        return self.minLevel <= z <= self.maxLevel

    def is_storing(self, z: int) -> bool:
        """True if a level goes to the persistent store."""

        return self.cache.maxAge > 0 and z <= self.cache.maxLevel

    ##

    def _get_tile_as_pair(self, mt: gws.MapTile, params: dict | None) -> _BlobImagePair:
        """Return a tile from the store, composing and storing its block on a miss."""

        z = mt[-1]
        if not self.is_serving(z):
            return self.empty_tile(), None

        if not gws.lib.grid.in_range(mt, self.tile_range_for_level(z)):
            return self.empty_tile(), None

        store_key = gws.u.sha256(params)[:12] if params else ''
        store = self._store_for(z, store_key)

        blob = store.read(mt)
        if blob is not None:
            return blob, None

        try:
            bx, by, _ = self.block_start_tile(mt)
            with gws.u.server_lock(f'grabber_{self.cache.name}_{store_key}_{z}_{bx}_{by}', BLOCK_LOCK_TIMEOUT):
                # the tile might be written by another render
                blob = store.read(mt)
                if blob is not None:
                    return blob, None
                block_images = self._compose_and_store_tile_block(mt, params, store)
                return None, block_images[mt]
        except gws.LockBusyError:
            gws.log.warning(f'grabber {self.cache.name!r}: block lock busy for {mt!r}')
            return self.empty_tile(), None

    def _compose_and_store_tile_block(self, mt: gws.MapTile, params: dict | None, store: gws.TileStore) -> dict[gws.MapTile, gws.Image]:
        """Compose the block containing a tile and write all its tiles to the store."""

        block_images = self.compose_tile_block_as_image_dict(mt, params)
        for t, img in block_images.items():
            blob = self.to_bytes(img)
            store.write(t, blob)
        if store is not self.store:
            gws.u.ephemeral_cleanup()
        return block_images

    def _get_box_as_pair(self, extent: gws.Extent, w: float, h: float, params: dict | None) -> _BlobImagePair:
        """Return a box, mosaicked from stored tiles at storing levels, composed directly otherwise."""

        w = gws.u.to_rounded_int(w)
        h = gws.u.to_rounded_int(h)

        z = gws.lib.grid.level_for_resolution(self.grid, gws.lib.extent.w(extent) / w)
        if params or not self.is_storing(z):
            return None, self.compose_box_as_image(extent, w, h, params)

        mtr = gws.lib.grid.range_for_extent(self.grid, extent, z)
        if not mtr:
            return self.empty_box(w, h), None

        x0, y0, x1, y1, _ = mtr
        ts = self.grid.tileSize
        mosaic = gws.lib.image.from_size(((x1 - x0 + 1) * ts, (y1 - y0 + 1) * ts))
        for (tx, ty, _), img in self.get_tiles_as_image_dict(mtr).items():
            mosaic.paste(img, ((tx - x0) * ts, (ty - y0) * ts))

        mosaic_extent = gws.lib.grid.extent_for_range(self.grid, mtr)
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

    def _valid_tiles_in_range(self, mtr: gws.MapTileRange) -> list[gws.MapTile]:
        """Return the tiles of a range that lie within the served levels and extent."""

        z = mtr[-1]
        if not self.is_serving(z):
            return []
        level_mtr = self.tile_range_for_level(z)
        return [t for t in gws.lib.grid.enum_tiles(mtr) if gws.lib.grid.in_range(t, level_mtr)]

    def _store_for(self, z: int, key: str = '') -> gws.TileStore:
        """Return the store for a level and params key."""

        if key:
            return self._ephemeral_store(key)
        if self.is_storing(z):
            return self.store
        return self.defaultEphemeralStore

    def _ephemeral_store(self, key: str) -> gws.TileStore:
        """Create an ephemeral store for a params key."""

        return gws.gis.cache.store.Object(
            gws.u.ephemeral_dir(f'tiles_{self.cache.name}_{key}'),
            max_age=EPHEMERAL_MAX_AGE,
            extension=gws.lib.mime.extension_for(self.mime),
        )
