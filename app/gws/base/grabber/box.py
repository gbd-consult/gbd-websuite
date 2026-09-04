"""Base grabber for boxed sources."""

import gws
import gws.lib.extent
import gws.lib.grid
import gws.lib.image

from . import core

DEFAULT_BLOCK_SIZE = 4
DEFAULT_EDGE_BUFFER = 64


class Config(core.Config):
    edgeBuffer: int
    """Pixel buffer around a tile block, rendered and cropped."""


class Object(core.Object):
    """Base grabber for sources that render arbitrary boxes."""

    edgeBuffer: int

    def configure(self):
        self.edgeBuffer = self.cfg('edgeBuffer') or DEFAULT_EDGE_BUFFER

    def fetch_tile(self, tile):
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

        img = self.fetch_box(extent, w, h)
        img.crop((buf, buf, w - buf, h - buf))
        arr = img.to_array()

        out = None
        for tx, ty in core.pairs(fx0, fx1, fy0, fy1):
            px = (tx - fx0) * ts
            py = (ty - fy0) * ts
            tile_img = gws.lib.image.from_array(arr[py:py + ts, px:px + ts].copy())
            blob = tile_img.to_bytes(self.mime, self.imageFormat.options)
            if (tx, ty) == (x, y):
                out = blob
            else:
                self.store_write((tx, ty, z), blob)
        return out
