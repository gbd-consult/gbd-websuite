"""Base grabber for tiled sources."""

import math

import gws
import gws.lib.extent
import gws.lib.grid
import gws.lib.image

from . import core


class Object(core.Object):
    """Base grabber for sources addressed as tile pyramids."""

    sourceMatrices: list[gws.TileMatrix]
    """Source tile matrices, coarsest first."""

    def compose_box_as_image(self, extent, w, h, params=None):
        """Mosaic the covering source tiles of the best matrix and warp them onto the box."""

        w = gws.u.to_rounded_int(w)
        h = gws.u.to_rounded_int(h)

        if self.targetCrs == self.sourceCrs:
            src_extent = extent
        else:
            src_extent = self.extent_to_source_crs(extent)
            if not src_extent:
                gws.log.debug(f'grabber {self.cache.name!r}: empty image: box {extent!r} outside the source CRS area')
                return self.empty_image(w, h)

        src_res = gws.lib.extent.w(src_extent) / w
        tm = self._matrix_for_resolution(src_res)
        if self.targetCrs != self.sourceCrs:
            src_extent = gws.lib.extent.buffer(src_extent, tm.resolution * 2)

        mtr = self._matrix_range_for_extent(tm, src_extent)
        if not mtr:
            gws.log.debug(f'grabber {self.cache.name!r}: empty image: box {extent!r} outside the source matrix {tm.identifier!r}')
            return self.empty_image(w, h)

        c0, r0, c1, r1, _ = mtr
        mw = (c1 - c0 + 1) * tm.tileWidth
        mh = (r1 - r0 + 1) * tm.tileHeight
        mosaic = gws.lib.image.from_size((mw, mh))

        for col, row, _ in gws.lib.grid.enum_tiles((c0, r0, c1, r1, 0)):
            img = self.fetch_tile_as_image(tm, col, row)
            px = (col - c0) * tm.tileWidth
            py = (row - r0) * tm.tileHeight
            mosaic.paste(img, (px, py))

        src_extent = self._matrix_extent_for_range(tm, mtr)
        return self.warp_image(mosaic, src_extent, extent, w, h)

    def fetch_tile_as_bytes(self, tm: gws.TileMatrix, col: int, row: int) -> bytes:
        """Fetch a source tile with exactly one request, as encoded bytes."""

        raise NotImplementedError(f'fetch_tile_as_bytes not implemented in {self!r}')

    def fetch_tile_as_image(self, tm: gws.TileMatrix, col: int, row: int) -> gws.Image:
        """Fetch a source tile with exactly one request, as an image."""

        raise NotImplementedError(f'fetch_tile_as_image not implemented in {self!r}')

    ##

    def _matrix_for_resolution(self, res: float) -> gws.TileMatrix:
        """Return the coarsest source matrix that does not need upscaling."""

        # Coarsest matrix with tm.res <= res, i.e. never upscale (downscale up to 2x).
        # Cross-CRS the wanted resolution rarely hits the source ladder, e.g. 3857 -> 25832
        # at 51N needs 1.6x the target resolution, landing between two levels.
        # Alternatives: nearest by ratio (upscale up to sqrt(2), coarser cartography, bigger labels)
        # or a threshold as in MapProxy (allow upscale below a factor, default 1.15).
        for tm in self.sourceMatrices:
            if tm.resolution <= res or math.isclose(tm.resolution, res):
                return tm
        return self.sourceMatrices[-1]

    def _matrix_range_for_extent(self, tm: gws.TileMatrix, extent: gws.Extent) -> gws.MapTileRange | None:
        """Return the range of matrix tiles covering an extent, or ``None`` outside the matrix."""

        tile_w = tm.resolution * tm.tileWidth
        tile_h = tm.resolution * tm.tileHeight

        # nudge the edges inward by a fraction of a tile, so that an edge lying exactly
        # on a tile boundary does not pull in a neighbouring tile through float noise
        epsilon = 1e-6

        x0 = math.floor((extent[0] - tm.x + tile_w * epsilon) / tile_w)
        x1 = math.floor((extent[2] - tm.x - tile_w * epsilon) / tile_w)
        y0 = math.floor((tm.y - extent[3] + tile_h * epsilon) / tile_h)
        y1 = math.floor((tm.y - extent[1] - tile_h * epsilon) / tile_h)

        if x0 <= x1 and y0 <= y1 and x1 >= 0 and y1 >= 0 and x0 < tm.width and y0 < tm.height:
            return (
                max(x0, 0),
                max(y0, 0),
                min(x1, int(tm.width) - 1),
                min(y1, int(tm.height) - 1),
                0,
            )

    def _matrix_extent_for_range(self, tm: gws.TileMatrix, mtr: gws.MapTileRange) -> gws.Extent:
        """Return the extent of a range of matrix tiles."""

        tile_w = tm.resolution * tm.tileWidth
        tile_h = tm.resolution * tm.tileHeight

        x0, y0, x1, y1, _ = mtr

        return (
            tm.x + x0 * tile_w,
            tm.y - (y1 + 1) * tile_h,
            tm.x + (x1 + 1) * tile_w,
            tm.y - y0 * tile_h,
        )
