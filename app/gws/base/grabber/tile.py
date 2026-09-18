"""Base grabber for tiled sources."""

import gws
import gws.lib.extent
import gws.lib.grid
import gws.lib.image

from . import core


class Object(core.Object):
    """Base grabber for sources addressed as tile pyramids."""

    sourceTms: gws.TileMatrixSet
    """Source tile matrix set."""

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
        tm = gws.lib.grid.matrix_for_resolution(self.sourceTms, src_res)
        if self.targetCrs != self.sourceCrs:
            src_extent = gws.lib.extent.buffer(src_extent, tm.resolution * 2)

        mtr = gws.lib.grid.matrix_range_for_extent(tm, src_extent)
        if not mtr:
            gws.log.debug(f'grabber {self.cache.name!r}: empty image: box {extent!r} outside the source matrix {tm.identifier!r}')
            return self.empty_image(w, h)

        x0, y0, x1, y1, _ = mtr
        mw = (x1 - x0 + 1) * tm.tileWidth
        mh = (y1 - y0 + 1) * tm.tileHeight
        mosaic = gws.lib.image.from_size((mw, mh))

        for x, y, _ in gws.lib.grid.enum_tiles((x0, y0, x1, y1, 0)):
            blob = self._fetch_and_cache_source_tile(tm, x, y)
            img = self.to_image(blob)
            px = (x - x0) * tm.tileWidth
            py = (y - y0) * tm.tileHeight
            mosaic.paste(img, (px, py))

        src_extent = gws.lib.grid.matrix_extent_for_range(tm, mtr)
        return self.warp_image(
            mosaic,
            gws.Bounds(crs=self.sourceCrs, extent=src_extent),
            gws.Bounds(crs=self.targetCrs, extent=extent),
            w,
            h,
        )

    def fetch_tile_as_bytes(self, tm: gws.TileMatrix, col: int, row: int) -> bytes:
        """Fetch a source tile with exactly one request, as encoded bytes."""

        raise NotImplementedError(f'fetch_tile_as_bytes not implemented in {self!r}')

    ##

    def _fetch_and_cache_source_tile(self, tm: gws.TileMatrix, col: int, row: int) -> bytes:
        """Return a source tile, fetched once and kept in the ephemeral store for neighbouring requests."""

        key = gws.u.sha256([self.cache.name, tm.identifier, col, row])
        return gws.u.get_ephemeral_content(
            f'source_tile_{key}',
            lambda: self.fetch_tile_as_bytes(tm, col, row),
        )
