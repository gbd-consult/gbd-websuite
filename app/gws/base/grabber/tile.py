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

    def compose_box_as_image(self, extent, width, height, params=None):
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        if self.targetCrs == self.sourceCrs:
            src_extent = extent
        else:
            src_extent = self.extent_to_source_crs(extent)
            if not src_extent:
                return gws.lib.image.from_size((w, h))

        mat = self.matrix_for_resolution((src_extent[2] - src_extent[0]) / w)
        if self.targetCrs != self.sourceCrs:
            src_extent = gws.lib.extent.buffer(src_extent, mat.resolution * 2)

        rng = self.matrix_range_for_extent(mat, src_extent)
        if not rng:
            return gws.lib.image.from_size((w, h))

        c0, r0, c1, r1, _ = rng
        mw = (c1 - c0 + 1) * mat.tileWidth
        mh = (r1 - r0 + 1) * mat.tileHeight
        mosaic = gws.lib.image.from_size((mw, mh))

        for col, row, _ in gws.lib.grid.enum_tiles((c0, r0, c1, r1, 0)):
            img = self.fetch_tile_as_image(mat, col, row)
            px = (col - c0) * mat.tileWidth
            py = (row - r0) * mat.tileHeight
            mosaic.paste(img, (px, py))

        src_extent = self.matrix_extent_for_range(mat, rng)
        return self.warp_image(mosaic, src_extent, extent, w, h)

    def fetch_tile_as_bytes(self, m: gws.TileMatrix, col: int, row: int) -> bytes:
        raise NotImplementedError(f'fetch_tile_as_bytes not implemented in {self!r}')

    def fetch_tile_as_image(self, m: gws.TileMatrix, col: int, row: int) -> gws.Image:
        return self.as_image((self.fetch_tile_as_bytes(m, col, row), None))

    def matrix_for_resolution(self, wanted: float) -> gws.TileMatrix:
        # Coarsest matrix with res <= wanted, i.e. never upscale (downscale up to 2x).
        # Cross-CRS the wanted resolution rarely hits the source ladder, e.g. 3857 -> 25832
        # at 51N needs 1.6x the target resolution, landing between two levels.
        # Alternatives: nearest by ratio (upscale up to sqrt(2), coarser cartography, bigger labels)
        # or a threshold as in MapProxy (allow upscale below a factor, default 1.15).
        for m in self.sourceMatrices:
            if m.resolution <= wanted * (1 + 1e-6):
                return m
        return self.sourceMatrices[-1]

    def matrix_range_for_extent(self, mat: gws.TileMatrix, extent: gws.Extent) -> gws.MapTileRange | None:
        tile_w = mat.resolution * mat.tileWidth
        tile_h = mat.resolution * mat.tileHeight

        c0 = math.floor((extent[0] - mat.x + tile_w * 1e-6) / tile_w)
        c1 = math.floor((extent[2] - mat.x - tile_w * 1e-6) / tile_w)
        r0 = math.floor((mat.y - extent[3] + tile_h * 1e-6) / tile_h)
        r1 = math.floor((mat.y - extent[1] - tile_h * 1e-6) / tile_h)

        if c1 < c0 or r1 < r0 or c1 < 0 or r1 < 0 or c0 >= mat.width or r0 >= mat.height:
            return None
        return (
            max(c0, 0),
            max(r0, 0),
            min(c1, int(mat.width) - 1),
            min(r1, int(mat.height) - 1),
            0,
        )

    def matrix_extent_for_range(self, mat: gws.TileMatrix, rng: gws.MapTileRange) -> gws.Extent:
        tile_w = mat.resolution * mat.tileWidth
        tile_h = mat.resolution * mat.tileHeight

        c0, r0, c1, r1, _ = rng

        return (
            mat.x + c0 * tile_w,
            mat.y - (r1 + 1) * tile_h,
            mat.x + (c1 + 1) * tile_w,
            mat.y - r0 * tile_h,
        )
