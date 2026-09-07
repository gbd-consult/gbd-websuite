"""Base grabber for boxed sources."""

import math

import gws
import gws.lib.extent
import gws.lib.grid
import gws.lib.gdalx
import gws.lib.image

from . import core


class Object(core.Object):
    """Base grabber for sources that render arbitrary boxes."""

    sourceCrs: gws.Crs
    requestTiles: int
    requestBuffer: int
    maxRequestPixels: int

    def configure(self):
        self.requestTiles = self.cache.requestTiles
        self.requestBuffer = self.cache.requestBuffer
        self.maxRequestPixels = 4096

    def fetch_tile(self, tile, params=None):
        x, y, z = tile
        n = 1 if params else self.requestTiles
        rng = self.rangeForLevel[z]

        fx0 = max((x // n) * n, rng[0])
        fy0 = max((y // n) * n, rng[1])
        fx1 = min((x // n) * n + n - 1, rng[2])
        fy1 = min((y // n) * n + n - 1, rng[3])

        ts = self.grid.tileSize
        res = gws.lib.grid.resolution_for_level(self.grid, z)
        buf = self.requestBuffer

        extent = gws.lib.grid.extent_for_range(self.grid, (fx0, fy0, fx1, fy1, z))
        extent = gws.lib.extent.buffer(extent, buf * res)

        w = (fx1 - fx0 + 1) * ts + 2 * buf
        h = (fy1 - fy0 + 1) * ts + 2 * buf

        img = self.draw_box(extent, w, h, params)
        img.crop((buf, buf, w - buf, h - buf))
        arr = img.to_array()

        out = None
        for tx, ty, _ in gws.lib.grid.enum_tiles((fx0, fy0, fx1, fy1, z)):
            px = (tx - fx0) * ts
            py = (ty - fy0) * ts
            tile_img = gws.lib.image.from_array(arr[py : py + ts, px : px + ts].copy())
            blob = tile_img.to_bytes(self.mime, self.imageFormat.options)
            if (tx, ty) == (x, y):
                out = blob
            else:
                self.store_write((tx, ty, z), blob, params)
        return out

    def draw_box(self, extent, width, height, params=None):
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        if self.sourceCrs == self.targetCrs:
            return self.draw_chunks(extent, w, h, params)

        src_extent = gws.lib.extent.transform(extent, self.targetCrs, self.sourceCrs)
        if not gws.lib.extent.is_valid(src_extent):
            return gws.lib.image.from_size((w, h))
        
        src_res = (src_extent[2] - src_extent[0]) / w
        src_extent = gws.lib.extent.buffer(src_extent, src_res * 2)
        sw = w + 4
        sh = h + 4

        img = self.draw_chunks(src_extent, sw, sh, params)

        with gws.lib.gdalx.open_from_image(img, gws.Bounds(crs=self.sourceCrs, extent=src_extent)) as ds:
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

    def draw_chunks(self, extent, w, h, params):
        mpx = self.maxRequestPixels

        if w <= mpx and h <= mpx:
            img = self.fetch_box(gws.Bounds(crs=self.sourceCrs, extent=extent), w, h, params)
            canvas = gws.lib.image.from_size((w, h))
            canvas.paste(img, (0, 0))
            return canvas

        buf = self.requestBuffer
        csize = mpx - 2 * buf
        xres = (extent[2] - extent[0]) / w
        yres = (extent[3] - extent[1]) / h

        canvas = gws.lib.image.from_size((w, h))

        for py in range(0, h, csize):
            for px in range(0, w, csize):
                cw = min(csize, w - px)
                ch = min(csize, h - py)
                e = (
                    extent[0] + (px - buf) * xres,
                    extent[3] - (py + ch + buf) * yres,
                    extent[0] + (px + cw + buf) * xres,
                    extent[3] - (py - buf) * yres,
                )
                img = self.fetch_box(gws.Bounds(crs=self.sourceCrs, extent=e), cw + 2 * buf, ch + 2 * buf, params)
                img.crop((buf, buf, buf + cw, buf + ch))
                canvas.paste(img, (px, py))

        return canvas

    def fetch_box(self, bounds: gws.Bounds, width: int, height: int, params: dict | None = None) -> gws.Image:
        raise NotImplementedError(f'fetch_box not implemented in {self!r}')
