"""Utilities to work with Bounds objects."""

from typing import Optional

import gws
import gws.gis.crs
import gws.gis.extent
import gws.gis.gml


def from_request_bbox(bbox: str, default_crs: gws.Crs = None, always_xy=False) -> Optional[gws.Bounds]:
    """Create Bounds from a KVP BBOX param.

    See OGC 06-121r9, 10.2.3 Bounding box KVP encoding.

    Args:
        bbox: A string with four coordinates, optionally followed by a CRS spec.
        default_crs: Default Crs.
        always_xy: If ``True``, coordinates are assumed to be in the XY (lon/lat) order

    Returns:
          A Bounds object.
    """

    if not bbox:
        return None

    crs = default_crs

    # x,y,x,y,crs
    ls = bbox.split(',')
    if len(ls) == 5:
        crs = gws.gis.crs.get(ls.pop())

    if not crs:
        return None

    extent = gws.gis.extent.from_list(ls)
    if not extent:
        return None

    return from_extent(extent, crs, always_xy)


def from_extent(extent: gws.Extent, crs: gws.Crs, always_xy=False) -> gws.Bounds:
    """Create Bounds from an Extent.

    Args:
        extent: An Extent.
        crs: A Crs object.
        always_xy: If ``True``, coordinates are assumed to be in the XY (lon/lat) order

    Returns:
          A Bounds object.
    """

    if crs.isYX and not always_xy:
        extent = gws.gis.extent.swap_xy(extent)

    return gws.Bounds(crs=crs, extent=extent)


def copy(b: gws.Bounds) -> gws.Bounds:
    """Copies and creates a new bounds object."""
    return gws.Bounds(crs=b.crs, extent=b.extent)


def union(bs: list[gws.Bounds]) -> gws.Bounds:
    """Creates the smallest bound that contains all the given bounds.

    Args:
        bs: Bounds.

    Returns:
        A Bounds object. Its crs is the same as the crs of the first object in bs.
    """

    crs = bs[0].crs
    exts = [gws.gis.extent.transform(b.extent, b.crs, crs) for b in bs]
    return gws.Bounds(
        crs=crs,
        extent=gws.gis.extent.union(exts))


def intersect(b1: gws.Bounds, b2: gws.Bounds) -> bool:
    """Returns ``True`` if the bounds are intersecting, otherwise ``False``."""
    e1 = b1.extent
    e2 = gws.gis.extent.transform(b2.extent, crs_from=b2.crs, crs_to=b1.crs)
    return gws.gis.extent.intersect(e1, e2)


def transform(b: gws.Bounds, crs_to: gws.Crs) -> gws.Bounds:
    """Transforms the bounds object to a different crs.

    Args:
        b: Bounds object.
        crs_to: Output crs.

    Returns:
        A bounds object.
    """
    if b.crs == crs_to:
        return b
    return gws.Bounds(
        crs=crs_to,
        extent=b.crs.transform_extent(b.extent, crs_to))


def buffer(b: gws.Bounds, buf: int) -> gws.Bounds:
    """Creates a bounds object with buffer to another bounds object.

    Args:
        b: A Bounds object.
        buf: Buffer between b and the output. If buf is positive the returned bounds object will be bigger.

    Returns:
        A bounds object.
    """
    if buf == 0:
        return b
    return gws.Bounds(crs=b.crs, extent=gws.gis.extent.buffer(b.extent, buf))
