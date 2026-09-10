"""Base grabber for tiled sources."""

import math

import gws
import gws.lib.extent
import gws.lib.gdalx
import gws.lib.grid
import gws.lib.image

from . import core


class Object(core.Object):
    """Base grabber for sources addressed as tile pyramids."""

    sourceCrs: gws.Crs
    sourceMatrices: list[gws.TileMatrix]

    def draw_box(self, extent, width, height, params=None):
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        src_extent = extent
        if self.targetCrs != self.sourceCrs:
            src_extent = gws.lib.extent.transform(src_extent, self.targetCrs, self.sourceCrs)
            if not gws.lib.extent.is_valid(src_extent):
                return gws.lib.image.from_size((w, h))

        m = self.matrix_for_resolution((src_extent[2] - src_extent[0]) / w)
        if self.targetCrs != self.sourceCrs:
            src_extent = gws.lib.extent.buffer(src_extent, matrix_resolution(m) * 2)

        rng = matrix_range(m, src_extent)
        if not rng:
            return gws.lib.image.from_size((w, h))

        c0, r0, c1, r1 = rng
        mw = (c1 - c0 + 1) * m.tileWidth
        mh = (r1 - r0 + 1) * m.tileHeight
        mosaic = gws.lib.image.from_size((mw, mh))

        for col, row, _ in gws.lib.grid.enum_tiles((c0, r0, c1, r1, 0)):
            blob = self.fetch_source_tile(m, col, row)
            ix = (col - c0) * m.tileWidth
            iy = (row - r0) * m.tileHeight
            img = gws.lib.image.from_bytes(blob)
            mosaic.paste(img, (ix, iy))

        src_bounds = gws.Bounds(crs=self.sourceCrs, extent=matrix_range_extent(m, rng))
        with gws.lib.gdalx.open_from_image(mosaic, src_bounds) as ds:
            img = ds.warp_to_image(dict(
                dstSRS=self.targetCrs.epsg,
                outputBounds=extent,
                outputBoundsSRS=self.targetCrs.epsg,
                width=w,
                height=h,
                resampleAlg='bilinear',
                warpOptions=['XSCALE=1', 'YSCALE=1'],
            ))

        return img

    def matrix_for_resolution(self, wanted: float) -> gws.TileMatrix:
        # Coarsest matrix with res <= wanted, i.e. never upscale (downscale up to 2x).
        # Cross-CRS the wanted resolution rarely hits the source ladder, e.g. 3857 -> 25832
        # at 51N needs 1.6x the target resolution, landing between two levels.
        # Alternatives: nearest by ratio (upscale up to sqrt(2), coarser cartography, bigger labels)
        # or a threshold as in MapProxy (allow upscale below a factor, default 1.15).
        for m in self.sourceMatrices:
            if matrix_resolution(m) <= wanted * (1 + 1e-6):
                return m
        return self.sourceMatrices[-1]

    def fetch_source_tile(self, m: gws.TileMatrix, col: int, row: int) -> bytes:
        raise NotImplementedError(f'fetch_source_tile not implemented in {self!r}')


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
