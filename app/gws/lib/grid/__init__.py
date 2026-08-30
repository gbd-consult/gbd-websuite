"""Map grids: tile pyramid math."""

import math
from typing import Optional

import gws

DEFAULT_TILE_SIZE = 256
MAX_LEVEL = 24

WEBMERCATOR_RADIUS = 6378137
WEBMERCATOR_SQUARE = (
    -math.pi * WEBMERCATOR_RADIUS,
    -math.pi * WEBMERCATOR_RADIUS,
    +math.pi * WEBMERCATOR_RADIUS,
    +math.pi * WEBMERCATOR_RADIUS,
)
GEOGRAPHIC_FRAME = (-180.0, -90.0, 180.0, 90.0)

BASE_RESOLUTION_PROJECTED = (WEBMERCATOR_SQUARE[2] - WEBMERCATOR_SQUARE[0]) / DEFAULT_TILE_SIZE
BASE_RESOLUTION_GEOGRAPHIC = (GEOGRAPHIC_FRAME[3] - GEOGRAPHIC_FRAME[1]) / DEFAULT_TILE_SIZE


class MapGridConfig(gws.Config):
    """Map grid options."""

    crs: Optional[gws.CrsName]
    extent: Optional[gws.Extent]
    baseResolution: Optional[float]
    tileSize: Optional[int]


class MapGridOptions(gws.Data):
    """Map grid options."""

    crs: gws.Crs
    extent: Optional[gws.Extent]
    baseResolution: Optional[float]
    tileSize: Optional[int]


def for_crs(crs: gws.Crs) -> gws.MapGrid:
    return new(MapGridOptions(crs=crs))


def new(opts: MapGridOptions) -> gws.MapGrid:
    mg = gws.MapGrid()
    mg.crs = opts.crs
    mg.extent = opts.extent or (GEOGRAPHIC_FRAME if mg.crs.isGeographic else WEBMERCATOR_SQUARE)
    mg.baseResolution = opts.baseResolution or (BASE_RESOLUTION_GEOGRAPHIC if mg.crs.isGeographic else BASE_RESOLUTION_PROJECTED)
    mg.tileSize = opts.tileSize or DEFAULT_TILE_SIZE
    return mg


def resolution_for_level(mg: gws.MapGrid, z: int) -> float:
    return mg.baseResolution / (1 << z)


def level_for_resolution(mg: gws.MapGrid, resolution: float) -> int:
    if resolution >= mg.baseResolution:
        return 0
    z = math.ceil(math.log2(mg.baseResolution / resolution) - 1e-9)
    return min(z, MAX_LEVEL)


def tile_count_for_level(mg: gws.MapGrid, z: int) -> tuple[int, int]:
    s = resolution_for_level(mg, z) * mg.tileSize
    return (
        max(1, round((mg.extent[2] - mg.extent[0]) / s)),
        max(1, round((mg.extent[3] - mg.extent[1]) / s)),
    )


def range_for_extent(mg: gws.MapGrid, extent: gws.Extent, z: int) -> gws.MapTileRange | None:
    f = mg.extent
    s = resolution_for_level(mg, z) * mg.tileSize
    nx, ny = tile_count_for_level(mg, z)
    eps = s * 1e-6

    x0 = math.floor((extent[0] - f[0] + eps) / s)
    x1 = math.floor((extent[2] - f[0] - eps) / s)
    y0 = math.floor((f[3] - extent[3] + eps) / s)
    y1 = math.floor((f[3] - extent[1] - eps) / s)

    if x1 < x0 or y1 < y0 or x1 < 0 or y1 < 0 or x0 >= nx or y0 >= ny:
        return None
    return max(x0, 0), max(y0, 0), min(x1, nx - 1), min(y1, ny - 1), z


def extent_for_range(mg: gws.MapGrid, tr: gws.MapTileRange) -> gws.Extent:
    x0, y0, x1, y1, z = tr
    f = mg.extent
    s = resolution_for_level(mg, z) * mg.tileSize
    return (
        f[0] + x0 * s,
        f[3] - (y1 + 1) * s,
        f[0] + (x1 + 1) * s,
        f[3] - y0 * s,
    )


def extent_for_tile(mg: gws.MapGrid, tile: gws.MapTile) -> gws.Extent:
    x, y, z = tile
    return extent_for_range(mg, (x, y, x, y, z))
