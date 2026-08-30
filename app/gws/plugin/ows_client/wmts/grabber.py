"""WMTS grabber."""

import math

import gws
import gws.base.grabber
import gws.lib.extent
import gws.lib.gdalx
import gws.lib.grid
import gws.lib.image

from . import provider


class Object(gws.base.grabber.Object):
    serviceProvider: provider.Object
    tms: gws.TileMatrixSet
    urlTemplate: str

    def configure(self):
        self.serviceProvider = self.cfg('_defaultProvider')
        self.tms = self.cfg('_defaultTms')
        self.urlTemplate = self.cfg('_defaultUrlTemplate')

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

        blob = self.get_box(
            gws.lib.grid.extent_for_tile(self.grid, tile),
            self.grid.tileSize,
            self.grid.tileSize,
        )
        self.store_write(tile, blob)

        return blob

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

        src_crs = self.tms.crs

        src_extent = extent
        if self.targetCrs != src_crs:
            src_extent = gws.lib.extent.transform(src_extent, self.targetCrs, src_crs)

        m = self.matrix_for_resolution((src_extent[2] - src_extent[0]) / w)
        if self.targetCrs != src_crs:
            src_extent = gws.lib.extent.buffer(src_extent, matrix_resolution(m) * 2)

        rng = matrix_range(m, src_extent)
        if not rng:
            return self.empty_box(w, h)

        c0, r0, c1, r1 = rng
        mw = (c1 - c0 + 1) * m.tileWidth
        mh = (r1 - r0 + 1) * m.tileHeight
        mosaic = gws.lib.image.from_size((mw, mh))

        for col, row in pairs(c0, c1, r0, r1):
            blob = self.serviceProvider.get_tile(self.urlTemplate, m.uid, col, row)
            ix = (col - c0) * m.tileWidth
            iy = (row - r0) * m.tileHeight
            img = gws.lib.image.from_bytes(blob)
            mosaic.paste(img, (ix, iy))

        src_bounds = gws.Bounds(crs=src_crs, extent=matrix_range_extent(m, rng))
        with gws.lib.gdalx.open_from_image(mosaic, src_bounds) as ds:
            img = ds.warp_to_image(dict(
                dstSRS=self.targetCrs.epsg,
                outputBounds=extent,
                outputBoundsSRS=self.targetCrs.epsg,
                width=w,
                height=h,
                resampleAlg='bilinear',
            ))

        return img.to_bytes(self.mime, self.imageFormat.options)

    def matrix_for_resolution(self, wanted: float) -> gws.TileMatrix:
        for m in self.tms.matrices:
            if matrix_resolution(m) <= wanted * (1 + 1e-6):
                return m
        return self.tms.matrices[-1]


def matrix_resolution(m: gws.TileMatrix) -> float:
    return (m.extent[2] - m.extent[0]) / (m.width * m.tileWidth)


def matrix_range(m: gws.TileMatrix, extent: gws.Extent) -> tuple[int, int, int, int] | None:
    res = matrix_resolution(m)
    sx = res * m.tileWidth
    sy = res * m.tileHeight

    c0 = math.floor((extent[0] - m.x + sx * 1e-6) / sx)
    c1 = math.floor((extent[2] - m.x - sx * 1e-6) / sx)
    r0 = math.floor((m.y - extent[3] + sy * 1e-6) / sy)
    r1 = math.floor((m.y - extent[1] - sy * 1e-6) / sy)

    if c1 < c0 or r1 < r0 or c1 < 0 or r1 < 0 or c0 >= m.width or r0 >= m.height:
        return None
    return max(c0, 0), max(r0, 0), min(c1, m.width - 1), min(r1, m.height - 1)


def matrix_range_extent(m: gws.TileMatrix, rng: tuple[int, int, int, int]) -> gws.Extent:
    res = matrix_resolution(m)
    sx = res * m.tileWidth
    sy = res * m.tileHeight
    c0, r0, c1, r1 = rng
    return (
        m.x + c0 * sx,
        m.y - (r1 + 1) * sy,
        m.x + (c1 + 1) * sx,
        m.y - r0 * sy,
    )


def pairs(x0, x1, y0, y1):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            yield x, y


def in_range(x, y, rng):
    return rng[0] <= x <= rng[2] and rng[1] <= y <= rng[3]

