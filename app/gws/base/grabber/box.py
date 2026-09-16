"""Base grabber for boxed sources."""

import math

import gws
import gws.lib.extent
import gws.lib.grid
import gws.lib.gdalx
import gws.lib.image

from . import core

MAX_SOURCE_PIXEL_RATIO = 16
"""Cap on source pixels per target pixel for cross-CRS requests."""


class Object(core.Object):
    """Base grabber for sources that render arbitrary boxes."""

    sourceCrs: gws.Crs
    requestBuffer: int
    maxRequestPixels: int

    def __init__(self, opts: core.Options):
        super().__init__(opts)
        self.requestTiles = self.cache.requestTiles
        self.requestBuffer = self.cache.requestBuffer
        self.maxRequestPixels = 4096

    def compose_tile_block_as_image_dict(self, tile, params=None):
        z = tile[-1]
        n = 1 if params else self.requestTiles
        bx, by, _ = tile if params else self.block_start_tile(tile)

        level_rng = self.tile_range_for_level(z)
        block_rng = (
            max(bx, level_rng[0]),
            max(by, level_rng[1]),
            min(bx + n - 1, level_rng[2]),
            min(by + n - 1, level_rng[3]),
            z,
        )

        tile_size = self.grid.tileSize
        buf_size = self.requestBuffer
        res = gws.lib.grid.resolution_for_level(self.grid, z)

        extent = gws.lib.grid.extent_for_range(self.grid, block_rng)
        extent = gws.lib.extent.buffer(extent, buf_size * res)

        w = (block_rng[2] - block_rng[0] + 1) * tile_size + 2 * buf_size
        h = (block_rng[3] - block_rng[1] + 1) * tile_size + 2 * buf_size

        img = self.compose_box_as_image(extent, w, h, params)
        img.crop((buf_size, buf_size, w - buf_size, h - buf_size))
        arr = img.to_array()

        tile_to_img = {}
        bx = block_rng[0]
        by = block_rng[1]

        for tx, ty, _ in gws.lib.grid.enum_tiles(block_rng):
            px = (tx - bx) * tile_size
            py = (ty - by) * tile_size
            slice = arr[py : py + tile_size, px : px + tile_size].copy()
            tile_to_img[(tx, ty, z)] = gws.lib.image.from_array(slice)

        return tile_to_img

    def compose_box_as_image(self, extent, width, height, params=None):
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        if self.sourceCrs == self.targetCrs:
            return self.compose_box_as_image_in_source_crs(extent, w, h, params)

        src_extent = self.extent_to_source_crs(extent)
        if not src_extent:
            return gws.lib.image.from_size((w, h))

        src_res = self.source_resolution(extent, (extent[2] - extent[0]) / w)
        if not src_res:
            return gws.lib.image.from_size((w, h))

        src_extent = gws.lib.extent.buffer(src_extent, src_res * 2)
        sw = math.ceil((src_extent[2] - src_extent[0]) / src_res)
        sh = math.ceil((src_extent[3] - src_extent[1]) / src_res)

        f = math.sqrt((sw * sh) / (w * h * MAX_SOURCE_PIXEL_RATIO))
        if f > 1:
            sw = math.ceil(sw / f)
            sh = math.ceil(sh / f)

        img = self.compose_box_as_image_in_source_crs(src_extent, sw, sh, params)
        return self.warp_image(img, src_extent, extent, w, h)

    def source_resolution(self, extent, res):
        tr = self.targetCrs.transformer(self.sourceCrs)
        x0, y0, x1, y1 = extent
        best = 0.0
        for x in (x0, (x0 + x1) / 2, x1):
            for y in (y0, (y0 + y1) / 2, y1):
                ax, ay = tr(x, y)
                bx, by = tr(x + res, y)
                cx, cy = tr(x, y + res)
                for d in (math.hypot(bx - ax, by - ay), math.hypot(cx - ax, cy - ay)):
                    if math.isfinite(d) and d > 0 and (not best or d < best):
                        best = d
        return best

    def compose_box_as_image_in_source_crs(self, extent, w, h, params):
        max_pix = self.maxRequestPixels

        if w <= max_pix and h <= max_pix:
            b = gws.Bounds(crs=self.sourceCrs, extent=extent)
            img = self.fetch_box_as_image(b, w, h, params)
            return self.normalize_image(img, w, h)

        buf_size = self.requestBuffer
        chunk_size = max_pix - 2 * buf_size
        xres = (extent[2] - extent[0]) / w
        yres = (extent[3] - extent[1]) / h

        canvas = gws.lib.image.from_size((w, h))

        for start_y in range(0, h, chunk_size):
            for start_x in range(0, w, chunk_size):
                cw = min(chunk_size, w - start_x)
                ch = min(chunk_size, h - start_y)
                fw = cw + 2 * buf_size
                fh = ch + 2 * buf_size
                e = (
                    extent[0] + (start_x - buf_size) * xres,
                    extent[3] - (start_y + ch + buf_size) * yres,
                    extent[0] + (start_x + cw + buf_size) * xres,
                    extent[3] - (start_y - buf_size) * yres,
                )
                img = self.fetch_box_as_image(gws.Bounds(crs=self.sourceCrs, extent=e), fw, fh, params)
                img = self.normalize_image(img, fw, fh)
                img.crop((buf_size, buf_size, buf_size + cw, buf_size + ch))
                canvas.paste(img, (start_x, start_y))

        return canvas

    def fetch_box_as_bytes(self, bounds: gws.Bounds, width: int, height: int, params: dict | None = None) -> bytes:
        raise NotImplementedError(f'fetch_box_as_bytes not implemented in {self!r}')

    def fetch_box_as_image(self, bounds: gws.Bounds, width: int, height: int, params: dict | None = None) -> gws.Image:
        raise NotImplementedError(f'fetch_box_as_image not implemented in {self!r}')
