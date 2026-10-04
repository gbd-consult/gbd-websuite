"""Bounds utilities.

A ``gws.Bounds`` object is an extent together with its CRS. This package
creates bounds from extents and OGC ``BBOX`` request parameters, transforms
them between CRS, and combines, compares and buffers them. Functions that
combine bounds in different CRS transform them to the CRS of the first
argument. Extents without a CRS are handled by ``gws.lib.extent``.

When bounds are created from external input, the axis order of the CRS is
respected: for a CRS with the YX (lat/lon) axis order, the coordinates are
swapped, unless the input is known to be in the XY order.

Example::

    b = gws.lib.bounds.from_request_bbox('7,50,8,51', gws.lib.crs.WGS84, always_xy=True)
    b = gws.lib.bounds.transform(b, gws.lib.crs.WEBMERCATOR)
    b = gws.lib.bounds.buffer(b, 100)
    wgs = gws.lib.bounds.wgs_extent(b)
"""

from typing import Optional

import gws
import gws.lib.crs
import gws.lib.extent
import gws.lib.gml

_PAD = 1e-3
_MIN_PAD = 1e-6


def from_request_bbox(bbox: str, default_crs: gws.Crs = None, always_xy=False) -> Optional[gws.Bounds]:
    """Create bounds from a KVP ``BBOX`` parameter.

    See OGC 06-121r9, 10.2.3 Bounding box KVP encoding.

    Args:
        bbox: Four comma-separated coordinates, optionally followed by a CRS name.
        default_crs: CRS to use if the parameter has no CRS.
        always_xy: If ``True``, coordinates are assumed to be in the XY (lon/lat) order.

    Returns:
        Bounds, or ``None`` if the parameter is empty or invalid, or there is no CRS.
    """

    if not bbox:
        return None

    crs = default_crs

    # x,y,x,y,crs
    ls = bbox.split(',')
    if len(ls) == 5:
        crs = gws.lib.crs.get(ls.pop())

    if not crs:
        return None

    extent = gws.lib.extent.from_list(ls)
    if not extent:
        return None

    return from_extent(extent, crs, always_xy)


def from_extent(extent: gws.Extent, crs: gws.Crs, always_xy=False) -> gws.Bounds:
    """Create bounds from an extent.

    If the CRS has the YX axis order, the extent coordinates are swapped, unless ``always_xy`` is set.

    Args:
        extent: Extent.
        crs: CRS of the extent.
        always_xy: If ``True``, coordinates are assumed to be in the XY (lon/lat) order.

    Returns:
        Bounds.
    """

    if crs.isYX and not always_xy:
        extent = gws.lib.extent.swap_xy(extent)

    return gws.Bounds(crs=crs, extent=extent)


def copy(b: gws.Bounds) -> gws.Bounds:
    """Copy bounds.

    Args:
        b: Bounds.

    Returns:
        New bounds with the same CRS and extent.
    """
    return gws.Bounds(crs=b.crs, extent=b.extent)


def union(bs: list[gws.Bounds]) -> gws.Bounds:
    """Create the smallest bounds that contain all given bounds.

    Args:
        bs: A non-empty list of bounds.

    Returns:
        Bounds in the CRS of the first element of ``bs``.
    """

    crs = bs[0].crs
    exts = [gws.lib.extent.transform(b.extent, b.crs, crs) for b in bs]
    return gws.Bounds(
        crs=crs,
        extent=gws.lib.extent.union(*exts),
    )


def intersect(b1: gws.Bounds, b2: gws.Bounds) -> bool:
    """Check if two bounds intersect.

    Args:
        b1: First bounds.
        b2: Second bounds, transformed to the CRS of ``b1`` for the check.

    Returns:
        ``True`` if the bounds intersect.
    """
    e1 = b1.extent
    e2 = gws.lib.extent.transform(b2.extent, crs_from=b2.crs, crs_to=b1.crs)
    return gws.lib.extent.intersect(e1, e2)


def transform(b: gws.Bounds, crs_to: gws.Crs) -> gws.Bounds:
    """Transform bounds to a different CRS.

    Args:
        b: Bounds.
        crs_to: Target CRS.

    Returns:
        Bounds in the target CRS, or ``b`` itself if it is already in that CRS.
    """
    if b.crs == crs_to:
        return b
    return gws.Bounds(
        crs=crs_to,
        extent=b.crs.transform_extent(b.extent, crs_to),
    )


def wgs_extent(b: gws.Bounds, pad: bool = False) -> Optional[gws.Extent]:
    """Transform bounds to a WGS84 extent.

    Args:
        b: Bounds.
        pad: Enlarge the extent slightly before the transformation, so that features on the edges
            of a data-derived extent survive the round trip through WGS84.

    Returns:
        A WGS84 extent, or ``None`` if the result is invalid.
    """

    ext = b.extent
    if pad:
        buf = max(gws.lib.extent.w(ext) * _PAD, gws.lib.extent.h(ext) * _PAD, _MIN_PAD)
        ext = gws.lib.extent.buffer(ext, buf)
    ext = gws.lib.extent.transform(ext, b.crs, gws.lib.crs.WGS84)
    return ext if gws.lib.extent.is_valid(ext) else None


def buffer(b: gws.Bounds, buf_size: float) -> gws.Bounds:
    """Enlarge or shrink bounds by a buffer.

    Args:
        b: Bounds.
        buf_size: Buffer size in CRS units. A positive buffer enlarges the bounds, a negative one shrinks them.

    Returns:
        New bounds, or ``b`` itself if the buffer is 0.
    """
    if buf_size == 0:
        return b
    return gws.Bounds(crs=b.crs, extent=gws.lib.extent.buffer(b.extent, buf_size))
