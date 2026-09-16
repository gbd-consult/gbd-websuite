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

    def compose_block_as_images(self, tile, params=None):
        x, y, z = tile
        n = 1 if params else self.requestTiles
        bx, by = (x // n) * n, (y // n) * n
        rng = self.rangeForLevel[z]

        fx0 = max(bx, rng[0])
        fy0 = max(by, rng[1])
        fx1 = min(bx + n - 1, rng[2])
        fy1 = min(by + n - 1, rng[3])

        ts = self.grid.tileSize
        res = gws.lib.grid.resolution_for_level(self.grid, z)
        buf = self.requestBuffer

        extent = gws.lib.grid.extent_for_range(self.grid, (fx0, fy0, fx1, fy1, z))
        extent = gws.lib.extent.buffer(extent, buf * res)

        w = (fx1 - fx0 + 1) * ts + 2 * buf
        h = (fy1 - fy0 + 1) * ts + 2 * buf

        img = self.compose_box_as_image(extent, w, h, params)
        img.crop((buf, buf, w - buf, h - buf))
        arr = img.to_array()

        images = {}
        for tx, ty, _ in gws.lib.grid.enum_tiles((fx0, fy0, fx1, fy1, z)):
            px = (tx - fx0) * ts
            py = (ty - fy0) * ts
            images[(tx, ty, z)] = gws.lib.image.from_array(arr[py : py + ts, px : px + ts].copy())
        return images

    def compose_box_as_image(self, extent, width, height, params=None):
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        if self.sourceCrs == self.targetCrs:
            return self.compose_chunks_as_image(extent, w, h, params)

        wgs_extent = self.sourceCrs.clip_extent(gws.lib.extent.transform_to_wgs(extent, self.targetCrs))
        if not wgs_extent:
            return gws.lib.image.from_size((w, h))

        src_extent = gws.lib.extent.transform_from_wgs(wgs_extent, self.sourceCrs)
        if not gws.lib.extent.is_valid(src_extent):
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

        img = self.compose_chunks_as_image(src_extent, sw, sh, params)

        with gws.lib.gdalx.open_from_image(img, gws.Bounds(crs=self.sourceCrs, extent=src_extent)) as ds:
            return ds.warp_to_image(
                dict(
                    dstSRS=self.targetCrs.epsg,
                    outputBounds=extent,
                    outputBoundsSRS=self.targetCrs.epsg,
                    width=w,
                    height=h,
                    resampleAlg='bilinear',
                    warpOptions=['XSCALE=1', 'YSCALE=1'],
                )
            )

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

    def compose_chunks_as_image(self, extent, w, h, params):
        mpx = self.maxRequestPixels

        if w <= mpx and h <= mpx:
            img = self.fetch_box_as_image(gws.Bounds(crs=self.sourceCrs, extent=extent), w, h, params)
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
                img = self.fetch_box_as_image(gws.Bounds(crs=self.sourceCrs, extent=e), cw + 2 * buf, ch + 2 * buf, params)
                img.crop((buf, buf, buf + cw, buf + ch))
                canvas.paste(img, (px, py))

        return canvas

    def fetch_box_as_bytes(self, bounds: gws.Bounds, width: int, height: int, params: dict | None = None) -> bytes:
        img = self.fetch_box_as_image(bounds, width, height, params)
        return img.to_bytes(self.mime, self.imageFormat.options)

    def fetch_box_as_image(self, bounds: gws.Bounds, width: int, height: int, params: dict | None = None) -> gws.Image:
        blob = self.fetch_box_as_bytes(bounds, width, height, params)
        return gws.lib.image.from_bytes(blob)
