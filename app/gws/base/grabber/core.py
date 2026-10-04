"""Base raster grabber."""

import math
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
    """Options for creating a grabber."""

    crs: gws.Crs
    """Target CRS."""
    cache: gws.MapCache
    """Cache settings, with the final cache name including the SRID."""
    extent: gws.Extent
    """Layer extent in the target CRS."""
    imageFormat: gws.ImageFormat
    """Format tiles are stored and returned in."""


class Object(gws.Grabber):
    """Base raster grabber.

    Implements the ``gws.Grabber`` API on top of the composition methods
    (``compose_*``), which subclasses implement for their kind of source.
    """

    defaultEphemeralStore: gws.TileStore
    """Short-lived store for static tiles outside the cache settings, so that blocks are composed once."""
    maxLevel: int
    """Finest served level."""
    mimeType: str
    """Mime type of ``imageFormat``."""
    minLevel: int
    """Coarsest served level."""
    mtrByLevel: dict[int, gws.MapTileRange]
    """Tile range covered by ``extent``, per served level."""
    requestTiles: int
    """Tiles per side composed in one block; 1 means no meta-tiling."""

    def __init__(self, opts: Options):
        """Create a grabber.

        Args:
            opts: Grabber options.

        Raises:
            ``gws.ConfigurationError``: If the extent covers no tiles at some level.
        """
        self.targetCrs = opts.crs
        self.grid = gws.lib.grid.for_crs(self.targetCrs)

        self.imageFormat = opts.imageFormat
        self.mimeType = self.imageFormat.mimeTypes[0]

        self.cache = opts.cache
        self.requestTiles = 1
        self.store = gws.gis.cache.store.Object(
            f'{gws.c.MAP_CACHE_DIR}/{self.cache.name}',
            max_age=self.cache.maxAge,
            extension=gws.lib.mime.extension_for(self.mimeType),
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
        """Compose the block of ``requestTiles`` x ``requestTiles`` tiles containing a tile.

        The base implementation supports only ``requestTiles`` 1 and composes the tile alone.

        Args:
            mt: A tile in the block.
            params: Dynamic request parameters.

        Returns:
            Images of the tiles of the block within the layer's tile range.

        Raises:
            ``NotImplementedError``: If ``requestTiles`` is not 1.
        """

        if self.requestTiles != 1:
            raise NotImplementedError(f'compose_tile_block_as_image_dict not implemented in {self!r}')
        return {mt: self.compose_tile_as_image(mt, params)}

    def compose_tile_as_image(self, mt: gws.MapTile, params: dict | None = None) -> gws.Image:
        """Compose a single tile over its own extent.

        Args:
            mt: The tile.
            params: Dynamic request parameters.

        Returns:
            The tile image.
        """

        return self.compose_box_as_image(
            gws.lib.grid.extent_for_tile(self.grid, mt),
            self.grid.tileSize,
            self.grid.tileSize,
            params,
        )

    def compose_box_as_image(self, extent: gws.Extent, w: int, h: int, params: dict | None = None) -> gws.Image:
        """Compose an image for an arbitrary extent and pixel size from the source.

        Subclasses must implement this.

        Args:
            extent: Extent in the target CRS.
            w: Width in pixels.
            h: Height in pixels.
            params: Dynamic request parameters.

        Returns:
            The image.

        Raises:
            ``NotImplementedError``: In the base class.
        """

        raise NotImplementedError(f'compose_box_as_image not implemented in {self!r}')

    ##

    def pair_to_bytes(self, bi: _BlobImagePair) -> bytes:
        """Return the bytes form of a pair, encoding the image if needed.

        Args:
            bi: A pair of encoded bytes and image, one of which is set.

        Returns:
            The encoded image.

        Raises:
            ``gws.Error``: If neither is set.
        """

        blob, img = bi
        if blob is not None:
            return blob
        if img is not None:
            return self.to_bytes(img)
        raise gws.Error('unexpected state')

    def pair_to_image(self, bi: _BlobImagePair) -> gws.Image:
        """Return the image form of a pair, decoding the bytes if needed.

        Args:
            bi: A pair of encoded bytes and image, one of which is set.

        Returns:
            The image.

        Raises:
            ``gws.Error``: If neither is set.
        """

        blob, img = bi
        if img is not None:
            return img
        if blob is not None:
            return self.to_image(blob)
        raise gws.Error('unexpected state')

    def to_bytes(self, img: gws.Image) -> bytes:
        """Encode an image in the grabber's image format.

        Args:
            img: The image.

        Returns:
            The encoded image.
        """

        return img.to_bytes(self.mimeType, self.imageFormat.options)

    def to_image(self, blob: bytes) -> gws.Image:
        """Decode an encoded image.

        Args:
            blob: The encoded image.

        Returns:
            The image.
        """

        return gws.lib.image.from_bytes(blob)

    def normalize_image(self, img: gws.Image, w: int, h: int) -> gws.Image:
        """Ensure a source image is RGBA and has the requested size.

        Args:
            img: Image from the source.
            w: Expected width.
            h: Expected height.

        Returns:
            The image converted to RGBA.

        Raises:
            ``gws.ExternalServiceError``: If the image size differs from the requested size.
        """

        if img.size() != (w, h):
            raise gws.ExternalServiceError(f'grabber {self.cache.name!r}: unexpected image size {img.size()!r}')
        return img.convert('RGBA')

    def block_start_tile(self, mt: gws.MapTile) -> gws.MapTile:
        """Return the first tile of the block containing a tile.

        Args:
            mt: The tile.

        Returns:
            The top-left tile of the block.
        """

        x, y, z = mt
        n = self.requestTiles
        return (x // n) * n, (y // n) * n, z

    def extent_to_source_crs(self, extent: gws.Extent) -> gws.Extent | None:
        """Transform a target extent into the source CRS, clipped to the source area of use.

        Args:
            extent: Extent in the target CRS.

        Returns:
            The extent in the source CRS, or ``None`` if it lies outside the source area of use.
        """

        wgs_extent = gws.lib.extent.transform_to_wgs(extent, self.targetCrs)
        wgs_extent = self.sourceCrs.clip_wgs_extent(wgs_extent)
        if not wgs_extent:
            return
        src_extent = gws.lib.extent.transform_from_wgs(wgs_extent, self.sourceCrs)
        if not gws.lib.extent.is_valid(src_extent):
            return
        return src_extent

    def warp_image(self, img: gws.Image, src_bounds: gws.Bounds, target_bounds: gws.Bounds, w: int, h: int) -> gws.Image:
        """Warp an image covering the source bounds onto the target bounds and pixel size.

        Uses nearest neighbour resampling when the CRS and the resolution are the same,
        and bilinear resampling otherwise.

        Args:
            img: Source image.
            src_bounds: Bounds covered by the source image.
            target_bounds: Target bounds.
            w: Target width in pixels.
            h: Target height in pixels.

        Returns:
            The warped image.
        """

        same_crs = src_bounds.crs == target_bounds.crs
        src_res = gws.lib.extent.w(src_bounds.extent) / img.size()[0]
        target_res = gws.lib.extent.w(target_bounds.extent) / w

        # Pixels copied 1:1 within the same CRS use 'nearest', so that a sub-pixel offset
        # between the source and target grids does not blur the image.
        resample_alg = 'bilinear'
        if same_crs and math.isclose(src_res, target_res, rel_tol=gws.lib.grid.RESOLUTION_TOLERANCE):
            resample_alg = 'nearest'

        # GDAL widens the bilinear kernel by the ratio of the source window to the destination chunk.
        # Cross-CRS, with a source covering e.g. the whole mercator world, the window estimate can fail
        # and fall back to the whole image, and the kernel then averages a strip of source pixels (diagonal smear).
        # Pinning the scale keeps a 2x2 kernel. Same-CRS the estimate is correct, so leave it alone.
        warp_options = [] if same_crs else ['XSCALE=1', 'YSCALE=1']

        with gws.lib.gdalx.open_from_image(img, src_bounds) as ds:
            return ds.warp_to_image(
                dict(
                    dstSRS=target_bounds.crs.epsg,
                    outputBounds=target_bounds.extent,
                    outputBoundsSRS=target_bounds.crs.epsg,
                    width=w,
                    height=h,
                    resampleAlg=resample_alg,
                    warpOptions=warp_options,
                )
            )

    def empty_image(self, w: int, h: int) -> gws.Image:
        """Return a transparent image of the given size.

        Args:
            w: Width in pixels.
            h: Height in pixels.

        Returns:
            The image.
        """

        return gws.lib.image.from_size((w, h))

    def empty_box(self, w: int, h: int) -> bytes:
        """Return a transparent image of the given size, as encoded bytes.

        Args:
            w: Width in pixels.
            h: Height in pixels.

        Returns:
            The encoded image.
        """

        return self.to_bytes(self.empty_image(w, h))

    def empty_tile(self) -> bytes:
        """Return the transparent tile, encoded once and then reused.

        Returns:
            The encoded tile.
        """

        if not hasattr(self, '_emptyTile'):
            self._emptyTile = self.empty_box(self.grid.tileSize, self.grid.tileSize)
        return self._emptyTile

    def is_serving(self, z: int) -> bool:
        """Check if a level is within the serving range.

        Args:
            z: Level.

        Returns:
            ``True`` if the level is served.
        """

        return self.minLevel <= z <= self.maxLevel

    def is_storing(self, z: int) -> bool:
        """Check if tiles of a level go to the persistent store.

        Args:
            z: Level.

        Returns:
            ``True`` if the cache has a positive ``maxAge`` and the level is not beyond ``cache.maxLevel``.
        """

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
        return None, self.warp_image(
            mosaic,
            gws.Bounds(crs=self.targetCrs, extent=mosaic_extent),
            gws.Bounds(crs=self.targetCrs, extent=extent),
            w,
            h,
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
            extension=gws.lib.mime.extension_for(self.mimeType),
        )
