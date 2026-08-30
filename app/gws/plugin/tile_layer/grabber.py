"""Tile layer grabber."""

import gws
import gws.base.grabber
import gws.lib.extent
import gws.lib.gdalx
import gws.lib.grid
import gws.lib.image

from . import provider


class Object(gws.base.grabber.Object):
    serviceProvider: provider.Object

    def configure(self):
        self.serviceProvider = self.cfg('_defaultProvider')

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

        sg = self.serviceProvider.grid

        src_extent = extent
        if self.targetCrs != sg.crs:
            src_extent = gws.lib.extent.transform(src_extent, self.targetCrs, sg.crs)

        z = gws.lib.grid.level_for_resolution(sg, (src_extent[2] - src_extent[0]) / w)
        z = min(z, self.serviceProvider.maxLevel)
        if self.targetCrs != sg.crs:
            src_res = gws.lib.grid.resolution_for_level(sg, z)
            src_extent = gws.lib.extent.buffer(src_extent, src_res * 2)

        src_rng = gws.lib.grid.range_for_extent(sg, src_extent, z)
        if not src_rng:
            return self.empty_box(w, h)

        sx0, sy0, sx1, sy1, _ = src_rng
        mw = (sx1 - sx0 + 1) * sg.tileSize
        mh = (sy1 - sy0 + 1) * sg.tileSize
        mosaic = gws.lib.image.from_size((mw, mh))

        for sx, sy in pairs(sx0, sx1, sy0, sy1):
            blob = self.serviceProvider.get_tile(sx, sy, z)
            ix = (sx - sx0) * sg.tileSize
            iy = (sy - sy0) * sg.tileSize
            img = gws.lib.image.from_bytes(blob)
            mosaic.paste(img, (ix, iy))

        src_bounds = gws.Bounds(crs=sg.crs, extent=gws.lib.grid.extent_for_range(sg, src_rng))
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


def pairs(x0, x1, y0, y1):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            yield x, y


def in_range(x, y, rng):
    return rng[0] <= x <= rng[2] and rng[1] <= y <= rng[3]

