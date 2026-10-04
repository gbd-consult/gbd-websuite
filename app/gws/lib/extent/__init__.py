"""Utilities for extents.

An extent (``gws.Extent``) is a tuple ``(x-min, y-min, x-max, y-max)`` of floats.
Extents have no CRS of their own; for a geo-referenced extent, see ``gws.Bounds`` and ``gws.lib.bounds``.

This package provides functions to:

- create extents from strings, lists, points, a center point and size, or a PostGIS ``BOX``,
- measure extents (center, size, width, height, diagonal),
- combine extents (intersection, union, buffer, circumscribed square),
- transform extents between CRSs,
- validate extents.

Example::

    import gws.lib.crs
    import gws.lib.extent

    ext = gws.lib.extent.from_string('0,0,100,50')
    gws.lib.extent.size(ext)  # (100.0, 50.0)
    gws.lib.extent.buffer(ext, 10)  # (-10.0, -10.0, 110.0, 60.0)

    wgs = gws.lib.extent.transform_to_wgs(ext, gws.lib.crs.WEBMERCATOR)
"""

from typing import Optional

import math
import re

import gws
import gws.lib.crs


def from_string(s: str) -> Optional[gws.Extent]:
    """Create an extent from a comma-separated string.

    Args:
        s: String like ``x-min,y-min,x-max,y-max``.

    Returns:
        An extent, or ``None`` if the string is not a valid extent.
    """

    return _from_string_list(s.split(','))


def from_list(ls: list) -> Optional[gws.Extent]:
    """Create an extent from a list.

    Args:
        ls: List ``[x-min, y-min, x-max, y-max]`` of numbers or numeric strings.

    Returns:
        An extent, or ``None`` if the list is not a valid extent.
    """

    return _from_string_list(ls)


def from_points(a: gws.Point, b: gws.Point) -> gws.Extent:
    """Create an extent from two opposite corner points.

    Args:
        a: First point.
        b: Second point.

    Returns:
        The smallest extent that contains both points.
    """

    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[0], b[0]),
        max(a[1], b[1]),
    )


def from_center(xy: gws.Point, size: gws.Size) -> gws.Extent:
    """Create an extent of a given size around a center point.

    Args:
        xy: Center point.
        size: Width and height.

    Returns:
        An extent.
    """

    return (
        xy[0] - size[0] / 2,
        xy[1] - size[1] / 2,
        xy[0] + size[0] / 2,
        xy[1] + size[1] / 2,
    )


def from_box(box: str) -> Optional[gws.Extent]:
    """Create an extent from a PostGIS box.

    Args:
        box: PostGIS box like ``BOX(minx miny,maxx maxy)``.

    Returns:
        An extent, or ``None`` if the box is empty or invalid.
    """

    if not box:
        return None

    m = re.match(r'^BOX\((.+?)\)$', str(box).upper())
    if not m:
        return None

    return _from_string_list(m.group(1).replace(',', ' ').split(' '))


#


def intersection(*exts: gws.Extent) -> Optional[gws.Extent]:
    """Create an extent that is the intersection of all given extents.

    Args:
        *exts: Extents.

    Returns:
        An extent, or ``None`` if the extents do not intersect or none are given.
    """

    if not exts:
        return

    res = (-math.inf, -math.inf, math.inf, math.inf)

    for ext in exts:
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
    """Get the center point of an extent.

    Args:
        e: An extent.

    Returns:
        The center point.
    """

    return (
        e[0] + (e[2] - e[0]) / 2,
        e[1] + (e[3] - e[1]) / 2,
    )


def size(e: gws.Extent) -> gws.Size:
    """Get the size of an extent.

    Args:
        e: An extent.

    Returns:
        A ``(width, height)`` tuple.
    """

    return (
        e[2] - e[0],
        e[3] - e[1],
    )


def w(e: gws.Extent) -> float:
    """Get the width of an extent.

    Args:
        e: An extent.

    Returns:
        The width.
    """

    return e[2] - e[0]


def h(e: gws.Extent) -> float:
    """Get the height of an extent.

    Args:
        e: An extent.

    Returns:
        The height.
    """

    return e[3] - e[1]


def diagonal(e: gws.Extent) -> float:
    """Get the length of the diagonal of an extent.

    Args:
        e: An extent.

    Returns:
        The diagonal length.
    """

    return math.sqrt((e[2] - e[0]) ** 2 + (e[3] - e[1]) ** 2)


def circumsquare(e: gws.Extent) -> gws.Extent:
    """Get the square that circumscribes an extent.

    The square has the same center as the extent, and its side equals the extent's diagonal,
    so that it contains the extent at any rotation.

    Args:
        e: An extent.

    Returns:
        The square extent.
    """

    d = diagonal(e)
    return from_center(center(e), (d, d))


def buffer(e: gws.Extent, buf: float) -> gws.Extent:
    """Create an extent with a buffer around another extent.

    Args:
        e: An extent.
        buf: Buffer added on each side. A positive value makes the extent bigger, a negative one smaller.

    Returns:
        The buffered extent.
    """

    if buf == 0:
        return e
    return (
        e[0] - buf,
        e[1] - buf,
        e[2] + buf,
        e[3] + buf,
    )


def union(*exts: gws.Extent) -> gws.Extent:
    """Create the smallest extent that contains all the given extents.

    Args:
        *exts: Extents, at least one.

    Returns:
        An extent.
    """

    ext = exts[0]
    for e in exts:
        ext = (
            min(ext[0], e[0]),
            min(ext[1], e[1]),
            max(ext[2], e[2]),
            max(ext[3], e[3]),
        )
    return ext


def intersect(a: gws.Extent, b: gws.Extent) -> bool:
    """Check if two extents intersect.

    Extents that only touch at the edges are considered intersecting.

    Args:
        a: First extent.
        b: Second extent.

    Returns:
        ``True`` if the extents intersect.
    """

    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def transform(e: gws.Extent, crs_from: gws.Crs, crs_to: gws.Crs) -> gws.Extent:
    """Transform an extent to a different coordinate reference system.

    Args:
        e: An extent.
        crs_from: Source CRS.
        crs_to: Target CRS.

    Returns:
        The transformed extent.
    """

    return crs_from.transform_extent(e, crs_to)


def transform_from_wgs(e: gws.Extent, crs_to: gws.Crs) -> gws.Extent:
    """Transform a WGS84 extent to a different coordinate reference system.

    Args:
        e: An extent in WGS84.
        crs_to: Target CRS.

    Returns:
        The transformed extent.
    """

    return gws.lib.crs.WGS84.transform_extent(e, crs_to)


def transform_to_wgs(e: gws.Extent, crs_from: gws.Crs) -> gws.Extent:
    """Transform an extent to WGS84.

    Args:
        e: An extent.
        crs_from: Source CRS.

    Returns:
        The WGS84 extent.
    """

    return crs_from.transform_extent(e, gws.lib.crs.WGS84)


def swap_xy(e: gws.Extent) -> gws.Extent:
    """Swap the x and y values of an extent.

    Args:
        e: An extent.

    Returns:
        The extent ``(y-min, x-min, y-max, x-max)``.
    """
    return e[1], e[0], e[3], e[2]


def is_valid(e: gws.Extent) -> bool:
    """Check if an extent is valid.

    A valid extent has four finite values, and its minimum values are less than its maximum values.

    Args:
        e: An extent.

    Returns:
        ``True`` if the extent is valid.
    """

    if not e or len(e) != 4:
        return False
    if not all(math.isfinite(p) for p in e):
        return False
    if e[0] >= e[2] or e[1] >= e[3]:
        return False
    return True


def is_valid_wgs(e: gws.Extent) -> bool:
    """Check if an extent is valid and lies within the WGS84 bounds.

    Args:
        e: An extent in WGS84.

    Returns:
        ``True`` if the extent is valid and within ``(-180, -90, 180, 90)``.
    """
    
    if not is_valid(e):
        return False
    w = gws.lib.crs.WGS84.extent
    return e[0] >= w[0] and e[1] >= w[1] and e[2] <= w[2] and e[3] <= w[3]


def _from_string_list(ls: list) -> Optional[gws.Extent]:
    if len(ls) != 4:
        return None
    try:
        e = [float(p) for p in ls]
    except ValueError:
        return None
    if not all(math.isfinite(p) for p in e):
        return None
    if e[0] >= e[2] or e[1] >= e[3]:
        return None
    return e[0], e[1], e[2], e[3]
