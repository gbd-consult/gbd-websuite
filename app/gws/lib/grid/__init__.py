"""Map grids: tile pyramid math.

A grid is a ``gws.MapGrid``: a CRS, a frame extent, a base resolution and a
tile size. The origin is the north-west corner of the frame. Level ``z`` has
the resolution ``baseResolution / 2**z``, so levels nest exactly (quad tree)
and the ladder has no bottom: any resolution can be served by downsampling
from the coarsest level whose resolution does not exceed it.

Defaults: projected CRS use the web mercator square as frame and one tile at
level 0; geographic CRS use ``(-180, -90, 180, 90)`` and two tiles (2x1) at
level 0. Without an explicit base resolution, one tile at level 0 spans the
frame height, so that the default resolutions are
``156543.03392804097 / 2**z`` metres and ``0.703125 / 2**z`` degrees for a
256px tile. Custom frames, base resolutions and tile sizes (tile source
grids) are possible; a non-square custom frame needs an explicit base
resolution. The frame is an indexing frame and does not have to be inside
the area of use of the CRS.

A custom grid can be snapped to the default grid of its CRS (``withSnap``):
the base resolution becomes a ladder value, the nearest one if given,
otherwise the coarsest whose tile spans the extent, and the extent is
expanded outward to tile boundaries at that level. A snapped grid is a
window on the default grid: its tile ``(x, y, z)`` is the default grid's
``(x + ox * 2**z, y + oy * 2**z, z + k)`` for the integer offset ``(ox, oy)``
of the window at level ``k``.

Tiles are ``(x, y, z)``, ranges ``(min_x, min_y, max_x, max_y, z)``.

Tile matrix sets (``gws.TileMatrixSet``, as in WMTS) describe foreign
pyramids: each matrix has its own origin, size and resolution, and the
resolutions need not form a ladder. ``matrix_set_for_grid`` expresses a grid
as a matrix set, the ``matrix_*`` functions are the matrix counterparts of
the grid functions.
"""

import math
from typing import Iterator, Optional

import gws
import gws.lib.crs

RESOLUTION_TOLERANCE = 0.01
"""Relative tolerance for a resolution to count as a level's resolution.

Requests are often rounded (bboxes with a few decimals), so a resolution slightly above
a level's must still map to that level rather than to the next finer one.
"""

DEFAULT_TILE_SIZE = 256


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
    withSnap: bool = True
    """Snap the extent and the base resolution to the default grid of the CRS."""


class Options(gws.Data):
    """Map grid options."""

    crs: gws.Crs
    extent: Optional[gws.Extent]
    baseResolution: Optional[float]
    tileSize: Optional[int]
    withSnap: Optional[bool]


def for_crs(crs: gws.Crs) -> gws.MapGrid:
    return new(Options(crs=crs))


def new(opts: Options) -> gws.MapGrid:
    mg = gws.MapGrid()
    mg.crs = opts.crs
    mg.extent = opts.extent or (gws.lib.crs.WGS84.extent if mg.crs.isGeographic else gws.lib.crs.WEBMERCATOR_SQUARE)
    mg.tileSize = opts.tileSize or DEFAULT_TILE_SIZE
    mg.baseResolution = opts.baseResolution or (mg.extent[3] - mg.extent[1]) / mg.tileSize
    with_snap = True if opts.withSnap is None else opts.withSnap
    if with_snap and (opts.extent or opts.baseResolution):
        _snap(mg, bool(opts.baseResolution))
    return mg


def _snap(mg: gws.MapGrid, has_base_resolution: bool):
    ref = for_crs(mg.crs)

    if has_base_resolution:
        k = round(math.log2(ref.baseResolution / mg.baseResolution))
    else:
        w = mg.extent[2] - mg.extent[0]
        h = mg.extent[3] - mg.extent[1]
        k = math.floor(math.log2(ref.baseResolution * mg.tileSize / max(w, h)) + 1e-9)

    mg.baseResolution = resolution_for_level(ref, max(k, 0))

    span = mg.baseResolution * mg.tileSize
    eps = span * 1e-6
    ox = ref.extent[0]
    oy = ref.extent[3]

    mg.extent = (
        ox + math.floor((mg.extent[0] - ox + eps) / span) * span,
        oy - math.ceil((oy - mg.extent[1] - eps) / span) * span,
        ox + math.ceil((mg.extent[2] - ox - eps) / span) * span,
        oy - math.floor((oy - mg.extent[3] + eps) / span) * span,
    )


def resolution_for_level(mg: gws.MapGrid, z: int) -> float:
    return mg.baseResolution / (1 << z)


def level_for_resolution(mg: gws.MapGrid, resolution: float) -> int:
    """Return the coarsest level whose resolution does not exceed the given one."""

    if resolution <= 0:
        raise ValueError(f'invalid resolution {resolution!r}')
    for z in range(100):
        r = resolution_for_level(mg, z)
        if r <= resolution or math.isclose(r, resolution, rel_tol=RESOLUTION_TOLERANCE):
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
        max(1, math.ceil((mg.extent[2] - mg.extent[0]) / span - 1e-6)),
        max(1, math.ceil((mg.extent[3] - mg.extent[1]) / span - 1e-6)),
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


def matrix_set_for_grid(mg: gws.MapGrid, max_level: int, identifier: str = '') -> gws.TileMatrixSet:
    """Express the levels ``0..max_level`` of a grid as a tile matrix set (matrix identifiers are the levels)."""

    tms = gws.TileMatrixSet(identifier=identifier, crs=mg.crs, matrices=[])
    for z in range(max_level + 1):
        nx, ny = tile_count_for_level(mg, z)
        tms.matrices.append(
            gws.TileMatrix(
                identifier=str(z),
                resolution=resolution_for_level(mg, z),
                x=mg.extent[0],
                y=mg.extent[3],
                width=nx,
                height=ny,
                tileWidth=mg.tileSize,
                tileHeight=mg.tileSize,
                extent=mg.extent,
            )
        )
    return tms


def matrix_for_resolution(tms: gws.TileMatrixSet, resolution: float) -> gws.TileMatrix:
    """Return the coarsest matrix that does not need upscaling, the finest one if all are coarser."""

    # Coarsest matrix with tm.resolution <= resolution, i.e. never upscale (downscale up to 2x).
    # Cross-CRS the wanted resolution rarely hits the source ladder, e.g. 3857 -> 25832
    # at 51N needs 1.6x the target resolution, landing between two levels.
    # Alternatives: nearest by ratio (upscale up to sqrt(2), coarser cartography, bigger labels)
    # or a threshold as in MapProxy (allow upscale below a factor, default 1.15).
    coarsest_first = sorted(tms.matrices, key=lambda m: -m.resolution)
    for tm in coarsest_first:
        if tm.resolution <= resolution or math.isclose(tm.resolution, resolution, rel_tol=RESOLUTION_TOLERANCE):
            return tm
    return coarsest_first[-1]


def matrix_range_for_extent(tm: gws.TileMatrix, extent: gws.Extent) -> gws.MapTileRange | None:
    """Return the range of matrix tiles covering an extent (``z`` is 0), or ``None`` outside the matrix."""

    tile_w = tm.resolution * tm.tileWidth
    tile_h = tm.resolution * tm.tileHeight

    # nudge the edges inward by a fraction of a tile, so that an edge lying exactly
    # on a tile boundary does not pull in a neighbouring tile through float noise
    eps = 1e-6

    x0 = math.floor((extent[0] - tm.x + tile_w * eps) / tile_w)
    x1 = math.floor((extent[2] - tm.x - tile_w * eps) / tile_w)
    y0 = math.floor((tm.y - extent[3] + tile_h * eps) / tile_h)
    y1 = math.floor((tm.y - extent[1] - tile_h * eps) / tile_h)

    if x1 < x0 or y1 < y0 or x1 < 0 or y1 < 0 or x0 >= tm.width or y0 >= tm.height:
        return None
    return max(x0, 0), max(y0, 0), min(x1, int(tm.width) - 1), min(y1, int(tm.height) - 1), 0


def matrix_extent_for_range(tm: gws.TileMatrix, tr: gws.MapTileRange) -> gws.Extent:
    """Return the extent of a range of matrix tiles."""

    tile_w = tm.resolution * tm.tileWidth
    tile_h = tm.resolution * tm.tileHeight
    x0, y0, x1, y1, _ = tr
    return (
        tm.x + x0 * tile_w,
        tm.y - (y1 + 1) * tile_h,
        tm.x + (x1 + 1) * tile_w,
        tm.y - y0 * tile_h,
    )


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
