"""WMS grabber."""

import gws
import gws.base.grabber
import gws.lib.extent
import gws.lib.gdalx
import gws.lib.grid
import gws.lib.image

from . import provider

DEFAULT_BLOCK_SIZE = 4
DEFAULT_EDGE_BUFFER = 64


class Object(gws.base.grabber.Object):
    serviceProvider: provider.Object
    sourceLayers: list[gws.SourceLayer]
    sourceCrs: gws.Crs
    edgeBuffer: int

    def configure(self):
        self.serviceProvider = self.cfg('_defaultProvider')
        self.sourceLayers = self.cfg('_defaultSourceLayers')
        self.sourceCrs = self.cfg('_defaultSourceCrs')
        self.edgeBuffer = self.cfg('edgeBuffer') or DEFAULT_EDGE_BUFFER

    ##

    def get_tile(self, tile):
        x, y, z = tile
        if not self.is_serving(z):
            return self.empty_tile()

        rng = self.rangeForLevel[z]
        if not in_range(x, y, rng):
            return self.empty_tile()

        blob = self.store_read(tile)
        if blob is not None:
            return blob

        return self.load_block(tile)

    def get_tiles(self, tr):
        x0, y0, x1, y1, z = tr
        if not self.is_serving(z):
            return {}

        rng = self.rangeForLevel[z]
        tiles = {}
        for x, y in pairs(x0, x1, y0, y1):
            if in_range(x, y, rng):
                tiles[x, y, z] = self.get_tile((x, y, z))
        return tiles

    def get_box(self, extent, width, height):
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        mpx = self.serviceProvider.maxRequestPixels
        rw = min(w, mpx)
        rh = min(h, mpx)

        if self.targetCrs == self.sourceCrs:
            blob = self.serviceProvider.get_map(
                gws.Bounds(crs=self.sourceCrs, extent=extent), rw, rh, self.sourceLayers, self.mime)
            if rw == w and rh == h:
                return blob
            img = gws.lib.image.from_bytes(blob)
            img.resize((w, h))
            return img.to_bytes(self.mime, self.imageFormat.options)

        src_extent = gws.lib.extent.transform(extent, self.targetCrs, self.sourceCrs)
        src_res = (src_extent[2] - src_extent[0]) / rw
        src_extent = gws.lib.extent.buffer(src_extent, src_res * 2)
        sw = rw + 4
        sh = rh + 4

        blob = self.serviceProvider.get_map(
            gws.Bounds(crs=self.sourceCrs, extent=src_extent), sw, sh, self.sourceLayers, self.mime)

        canvas = gws.lib.image.from_size((sw, sh))
        canvas.paste(gws.lib.image.from_bytes(blob), (0, 0))

        with gws.lib.gdalx.open_from_image(canvas, gws.Bounds(crs=self.sourceCrs, extent=src_extent)) as ds:
            img = ds.warp_to_image(dict(
                dstSRS=self.targetCrs.epsg,
                outputBounds=extent,
                outputBoundsSRS=self.targetCrs.epsg,
                width=w,
                height=h,
                resampleAlg='bilinear',
            ))

        return img.to_bytes(self.mime, self.imageFormat.options)

    ##

    def load_block(self, tile):
        x, y, z = tile
        n = self.blockSize
        rng = self.rangeForLevel[z]

        fx0 = max((x // n) * n, rng[0])
        fy0 = max((y // n) * n, rng[1])
        fx1 = min((x // n) * n + n - 1, rng[2])
        fy1 = min((y // n) * n + n - 1, rng[3])

        ts = self.grid.tileSize
        res = gws.lib.grid.resolution_for_level(self.grid, z)
        buf = self.edgeBuffer

        extent = gws.lib.grid.extent_for_range(self.grid, (fx0, fy0, fx1, fy1, z))
        extent = gws.lib.extent.buffer(extent, buf * res)

        w = (fx1 - fx0 + 1) * ts + 2 * buf
        h = (fy1 - fy0 + 1) * ts + 2 * buf

        blob = self.get_box(extent, w, h)

        canvas = gws.lib.image.from_size((w, h))
        canvas.paste(gws.lib.image.from_bytes(blob), (0, 0))
        canvas.crop((buf, buf, w - buf, h - buf))
        arr = canvas.to_array()

        out = None
        for tx, ty in pairs(fx0, fx1, fy0, fy1):
            px = (tx - fx0) * ts
            py = (ty - fy0) * ts
            img = gws.lib.image.from_array(arr[py:py + ts, px:px + ts].copy())
            b = img.to_bytes(self.mime, self.imageFormat.options)
            self.store_write((tx, ty, z), b)
            if (tx, ty) == (x, y):
                out = b
        return out


def pairs(x0, x1, y0, y1):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            yield x, y


def in_range(x, y, rng):
    return rng[0] <= x <= rng[2] and rng[1] <= y <= rng[3]

