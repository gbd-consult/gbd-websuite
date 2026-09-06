"""Map grids: tile pyramid math."""

import math
from typing import Iterator, Optional

import gws

DEFAULT_TILE_SIZE = 256

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

class Props(gws.Props):
    origin: str
    extent: gws.Extent
    resolutions: list[float]
    tileSize: int




class Config(gws.Config):
    """Map grid options."""

    crs: Optional[gws.CrsName]
    extent: Optional[gws.Extent]
    baseResolution: Optional[float]
    tileSize: Optional[int]


class Options(gws.Data):
    """Map grid options."""

    crs: gws.Crs
    extent: Optional[gws.Extent]
    baseResolution: Optional[float]
    tileSize: Optional[int]


def for_crs(crs: gws.Crs) -> gws.MapGrid:
    return new(Options(crs=crs))


def new(opts: Options) -> gws.MapGrid:
    mg = gws.MapGrid()
    mg.crs = opts.crs
    mg.extent = opts.extent or (GEOGRAPHIC_FRAME if mg.crs.isGeographic else WEBMERCATOR_SQUARE)
    mg.baseResolution = opts.baseResolution or (BASE_RESOLUTION_GEOGRAPHIC if mg.crs.isGeographic else BASE_RESOLUTION_PROJECTED)
    mg.tileSize = opts.tileSize or DEFAULT_TILE_SIZE
    return mg


def resolution_for_level(mg: gws.MapGrid, z: int) -> float:
    return mg.baseResolution / (1 << z)


def level_for_resolution(mg: gws.MapGrid, resolution: float) -> int:
    if resolution <= 0:
        raise ValueError(f'invalid resolution {resolution!r}')
    for z in range(100):
        r = resolution_for_level(mg, z)
        if r <= resolution or math.isclose(r, resolution):
            return z
    raise ValueError(f'invalid resolution {resolution!r}')


def props_for_resolutions(mg: gws.MapGrid, resolutions: list[float]) -> Props:
    zmax = level_for_resolution(mg, min(resolutions))
    return Props(
        origin=gws.Origin.nw,
        extent=mg.extent,
        resolutions=[resolution_for_level(mg, z) for z in range(zmax + 1)],
        tileSize=mg.tileSize,
    )


def tile_count_for_level(mg: gws.MapGrid, z: int) -> tuple[int, int]:
    span = resolution_for_level(mg, z) * mg.tileSize
    return (
        max(1, round((mg.extent[2] - mg.extent[0]) / span)),
        max(1, round((mg.extent[3] - mg.extent[1]) / span)),
    )


def range_for_extent(mg: gws.MapGrid, extent: gws.Extent, z: int) -> gws.MapTileRange | None:
    span = resolution_for_level(mg, z) * mg.tileSize
    nx, ny = tile_count_for_level(mg, z)
    eps = span * 1e-6

    x0 = math.floor((extent[0] - mg.extent[0] + eps) / span)
    x1 = math.floor((extent[2] - mg.extent[0] - eps) / span)
    y0 = math.floor((mg.extent[3] - extent[3] + eps) / span)
    y1 = math.floor((mg.extent[3] - extent[1] - eps) / span)

    if x1 < x0 or y1 < y0 or x1 < 0 or y1 < 0 or x0 >= nx or y0 >= ny:
        return None
    return max(x0, 0), max(y0, 0), min(x1, nx - 1), min(y1, ny - 1), z


def extent_for_range(mg: gws.MapGrid, tr: gws.MapTileRange) -> gws.Extent:
    x0, y0, x1, y1, z = tr
    span = resolution_for_level(mg, z) * mg.tileSize
    return (
        mg.extent[0] + x0 * span,
        mg.extent[3] - (y1 + 1) * span,
        mg.extent[0] + (x1 + 1) * span,
        mg.extent[3] - y0 * span,
    )


def extent_for_tile(mg: gws.MapGrid, tile: gws.MapTile) -> gws.Extent:
    x, y, z = tile
    return extent_for_range(mg, (x, y, x, y, z))


def enum_tiles(tr: gws.MapTileRange) -> Iterator[gws.MapTile]:
    """Enumerate the tiles of a range, row by row."""

    x0, y0, x1, y1, z = tr
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            yield x, y, z


def in_range(mt: gws.MapTile, tr: gws.MapTileRange) -> bool:
    """True if the tile lies within the range."""

    x, y, z = mt
    return z == tr[4] and tr[0] <= x <= tr[2] and tr[1] <= y <= tr[3]
