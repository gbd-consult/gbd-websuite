"""Base grabber for boxed sources."""

import math

import gws
import gws.lib.extent
import gws.lib.grid
import gws.lib.image

from . import core

MAX_SOURCE_PIXEL_RATIO = 4
"""Cap on source pixels per target pixel per side, for cross-CRS requests."""


class Object(core.Object):
    """Base grabber for sources that render arbitrary boxes."""

    maxRequestPixels: int
    """Cap on the pixel size of one source request; larger boxes are fetched in chunks."""
    requestBuffer: int
    """Pixels rendered around a block or chunk and cropped, for consistent labels across seams."""

    def __init__(self, opts: core.Options):
        super().__init__(opts)
        self.requestTiles = self.cache.requestTiles
        self.requestBuffer = self.cache.requestBuffer
        self.maxRequestPixels = 4096

    def compose_tile_block_as_image_dict(self, mt, params=None):
        """Render the block plus a buffer in one request and cut it into tiles."""

        z = mt[-1]
        n = self.requestTiles
        bx, by, _ = self.block_start_tile(mt)

        level_mtr = self.tile_range_for_level(z)
        lx0, ly0, lx1, ly1, _ = level_mtr

        block_mtr = (
            max(bx, lx0),
            max(by, ly0),
            min(bx + n - 1, lx1),
            min(by + n - 1, ly1),
            z,
        )
        bx0, by0, bx1, by1, _ = block_mtr

        tile_size = self.grid.tileSize
        buf_size = self.requestBuffer
        res = gws.lib.grid.resolution_for_level(self.grid, z)

        extent = gws.lib.grid.extent_for_range(self.grid, block_mtr)
        extent = gws.lib.extent.buffer(extent, buf_size * res)

        w = (bx1 - bx0 + 1) * tile_size + buf_size * 2
        h = (by1 - by0 + 1) * tile_size + buf_size * 2

        img = self.compose_box_as_image(extent, w, h, params)
        img.crop((buf_size, buf_size, w - buf_size, h - buf_size))
        pixels = img.to_array()

        block_images = {}

        for tx, ty, _ in gws.lib.grid.enum_tiles(block_mtr):
            px = (tx - bx0) * tile_size
            py = (ty - by0) * tile_size
            slice = pixels[py : py + tile_size, px : px + tile_size]
            block_images[(tx, ty, z)] = gws.lib.image.from_array(slice.copy())

        return block_images

    def compose_box_as_image(self, extent, w, h, params=None):
        """Fetch the box in the source CRS, warping it when the CRS differ."""

        w = gws.u.to_rounded_int(w)
        h = gws.u.to_rounded_int(h)

        if self.sourceCrs == self.targetCrs:
            return self._fetch_and_compose_box(extent, w, h, params)

        src_extent = self.extent_to_source_crs(extent)
        if not src_extent:
            gws.log.debug(f'grabber {self.cache.name!r}: empty image: box {extent!r} outside the source CRS area')
            return self.empty_image(w, h)

        target_res = gws.lib.extent.w(extent) / w
        src_res = self.targetCrs.transform_resolution(extent, target_res, self.sourceCrs)
        if not src_res:
            gws.log.debug(f'grabber {self.cache.name!r}: empty image: no source resolution for box {extent!r}')
            return self.empty_image(w, h)

        src_extent = gws.lib.extent.buffer(src_extent, src_res * 2)
        sw = math.ceil(gws.lib.extent.w(src_extent) / src_res)
        sh = math.ceil(gws.lib.extent.h(src_extent) / src_res)

        # src_res is the finest resolution anywhere in the box, so where the projection is strongly
        # distorted the source request can grow far beyond the output. Cap it at N times the output
        # per side, shrinking both sides by the same factor to keep the source pixels square.
        factor = max(
            sw / (w * MAX_SOURCE_PIXEL_RATIO),
            sh / (h * MAX_SOURCE_PIXEL_RATIO),
        )
        if factor > 1:
            sw = math.ceil(sw / factor)
            sh = math.ceil(sh / factor)

        img = self._fetch_and_compose_box(src_extent, sw, sh, params)
        return self.warp_image(img, src_extent, extent, w, h)

    def fetch_box_as_bytes(self, bounds: gws.Bounds, w: int, h: int, params: dict | None = None) -> bytes:
        """Fetch a box from the source with exactly one request, as encoded bytes."""

        raise NotImplementedError(f'fetch_box_as_bytes not implemented in {self!r}')

    def fetch_box_as_image(self, bounds: gws.Bounds, w: int, h: int, params: dict | None = None) -> gws.Image:
        """Fetch a box from the source with exactly one request, as an image."""

        raise NotImplementedError(f'fetch_box_as_image not implemented in {self!r}')

    ##

    def _fetch_and_compose_box(self, extent: gws.Extent, w: int, h: int, params: dict | None) -> gws.Image:
        """Fetch a source-CRS box, split into chunks of at most ``maxRequestPixels``."""

        max_pix = self.maxRequestPixels

        if w <= max_pix and h <= max_pix:
            b = gws.Bounds(crs=self.sourceCrs, extent=extent)
            img = self.fetch_box_as_image(b, w, h, params)
            return self.normalize_image(img, w, h)

        buf_size = self.requestBuffer
        chunk_size = max_pix - buf_size * 2
        
        x_res = gws.lib.extent.w(extent) / w
        y_res = gws.lib.extent.h(extent) / h

        canvas = gws.lib.image.from_size((w, h))

        for sy in range(0, h, chunk_size):
            for sx in range(0, w, chunk_size):
                chunk_w = min(chunk_size, w - sx)
                chunk_h = min(chunk_size, h - sy)
                fetch_w = chunk_w + buf_size * 2
                fetch_h = chunk_h + buf_size * 2
                e = (
                    extent[0] + (sx - buf_size) * x_res,
                    extent[3] - (sy + chunk_h + buf_size) * y_res,
                    extent[0] + (sx + chunk_w + buf_size) * x_res,
                    extent[3] - (sy - buf_size) * y_res,
                )
                img = self.fetch_box_as_image(
                    gws.Bounds(crs=self.sourceCrs, extent=e),
                    fetch_w,
                    fetch_h,
                    params,
                )
                img = self.normalize_image(img, fetch_w, fetch_h)
                img.crop(
                    (
                        buf_size,
                        buf_size,
                        buf_size + chunk_w,
                        buf_size + chunk_h,
                    )
                )
                canvas.paste(img, (sx, sy))

        return canvas
