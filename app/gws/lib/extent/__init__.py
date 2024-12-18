from typing import Optional

import math
import re

import gws
import gws.lib.crs


def from_string(s: str) -> Optional[gws.Extent]:
    """Create an extent from a comma-separated string "1000,2000,20000 40000".

    Args:
        s: ``"x-min,y-min,x-max,y-max"``

    Returns:
        An extent.
    """

    return from_list(s.split(','))


def from_list(ls: list) -> Optional[gws.Extent]:
    """Create an extent from a list of values.

    Args:
        ls: ``[x-min,y-min,x-max,y-max]``

    Returns:
        An extent."""

    return _check(ls)


def from_points(a: gws.Point, b: gws.Point) -> gws.Extent:
    """Create an extent from two points.

        Args:
            a:``(x-min,y-min)``
            b:``(x-max,y-max)``

        Returns:
            An extent."""

    return _check([a[0], a[1], b[0], b[1]])


def from_center(xy: gws.Point, size: gws.Size) -> gws.Extent:
    """Create an extent with certain size from a center-point.

        Args:
            xy: Center-point ``(x,y)``
            size: Extent's size.

        Returns:
            An Extent."""

    return (
        xy[0] - size[0] / 2,
        xy[1] - size[1] / 2,
        xy[0] + size[0] / 2,
        xy[1] + size[1] / 2,
    )


def from_box(box: str) -> Optional[gws.Extent]:
    """Create an extent from a Postgis BOX(1000 2000,20000 40000).

    Args:
        box: Postgis BOX.

    Returns:
        An extent."""

    if not box:
        return None

    m = re.match(r'^BOX\((.+?)\)$', str(box).upper())
    if not m:
        return None

    a, b = m.group(1).split(',')
    c, d = a.split(), b.split()

    return _check([c[0], c[1], d[0], d[1]])


#

def intersection(exts: list[gws.Extent]) -> Optional[gws.Extent]:
    """Creates an extent that is the intersection of all given extents.

    Args:
        exts: Extents.

    Returns:
        An extent.
    """

    if not exts:
        return

    res = (-math.inf, -math.inf, math.inf, math.inf)

    for ext in exts:
        _sort(ext)
        if not intersect(res, ext):
            return
        res = (
            max(res[0], ext[0]),
            max(res[1], ext[1]),
            min(res[2], ext[2]),
            min(res[3], ext[3]),
        )
    return res


def center(e: gws.Extent) -> gws.Point:
    """The center-point of the extent"""

    return (
        e[0] + (e[2] - e[0]) / 2,
        e[1] + (e[3] - e[1]) / 2,
    )


def size(e: gws.Extent) -> gws.Size:
    """The size of the extent ``(width,height)"""

    return (
        e[2] - e[0],
        e[3] - e[1],
    )


def diagonal(e: gws.Extent) -> float:
    """The length of the diagonal"""

    return math.sqrt((e[2] - e[0]) ** 2 + (e[3] - e[1]) ** 2)


def circumsquare(e: gws.Extent) -> gws.Extent:
    """A circumscribed square of the extent."""

    d = diagonal(e)
    return from_center(center(e), (d, d))


def buffer(e: gws.Extent, buf: int) -> gws.Extent:
    """Creates an extent with buffer to another extent.

    Args:
        e: An extent.
        buf: Buffer between e and the output. If buf is positive the returned extent will be bigger.

    Returns:
        An extent.
    """

    if buf == 0:
        return e
    e = _sort(e)
    return (
        e[0] - buf,
        e[1] - buf,
        e[2] + buf,
        e[3] + buf,
    )


def union(exts: list[gws.Extent]) -> gws.Extent:
    """Creates the smallest extent that contains all the given extents.

    Args:
        exts: Extents.

    Returns:
        An Extent.
    """

    ext = exts[0]
    for e in exts:
        e = _sort(e)
        ext = (
            min(ext[0], e[0]),
            min(ext[1], e[1]),
            max(ext[2], e[2]),
            max(ext[3], e[3])
        )
    return ext


def intersect(a: gws.Extent, b: gws.Extent) -> bool:
    """Returns ``True`` if the extents are intersecting, otherwise ``False``."""

    a = _sort(a)
    b = _sort(b)
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def transform(e: gws.Extent, crs_from: gws.Crs, crs_to: gws.Crs) -> gws.Extent:
    """Transforms the extent to a different coordinate reference system.

    Args:
        e: An extent.
        crs_from: Input crs.
        crs_to: Output crs.

    Returns:
        The transformed extent.
    """

    return crs_from.transform_extent(e, crs_to)


def transform_from_wgs(e: gws.Extent, crs_to: gws.Crs) -> gws.Extent:
    """Transforms the extent in WGS84 to a different coordinate reference system.

    Args:
        e: An extent.
        crs_to: Output crs.

    Returns:
        The transformed extent.
    """

    return gws.lib.crs.WGS84.transform_extent(e, crs_to)


def transform_to_wgs(e: gws.Extent, crs_from: gws.Crs) -> gws.Extent:
    """Transforms the extent to WGS84.

    Args:
        e: An extent.
        crs_from: Input crs.

    Returns:
        The WGS84 extent.
    """

    return crs_from.transform_extent(e, gws.lib.crs.WGS84)


def swap_xy(e: gws.Extent) -> gws.Extent:
    """Swaps the x and y values of the extent"""
    return e[1], e[0], e[3], e[2]


def is_valid(e: gws.Extent) -> bool:
    if not e or len(e) != 4:
        return False
    if not all(math.isfinite(p) for p in e):
        return False
    if e[0] >= e[2] or e[1] >= e[3]:
        return False
    return True


def is_valid_wgs(e: gws.Extent) -> bool:
    if not is_valid(e):
        return False
    w = gws.lib.crs.WGS84.extent
    return e[0] >= w[0] and e[1] >= w[1] and e[2] <= w[2] and e[3] <= w[3]


def _check(ls: list) -> Optional[gws.Extent]:
    if len(ls) != 4:
        return None
    try:
        e = [float(p) for p in ls]
    except ValueError:
        return None
    if not all(math.isfinite(p) for p in e):
        return None
    e = _sort(e)
    if e[0] >= e[2] or e[1] >= e[3]:
        return None
    return e


def _sort(e):
    # our extents are always [minx, miny, maxx, maxy]
    return (
        min(e[0], e[2]),
        min(e[1], e[3]),
        max(e[0], e[2]),
        max(e[1], e[3]),
    )
