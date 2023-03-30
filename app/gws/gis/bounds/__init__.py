"""Utilities to work with Bounds objects."""

import gws
import gws.gis.crs
import gws.gis.extent
import gws.gis.gml
import gws.types as t


def from_request_bbox(bbox: str, default_crs: gws.ICrs = None, always_xy=False) -> t.Optional[gws.Bounds]:
    """Create Bounds from a KVP BBOX param.

    See OGC 06-121r9, 10.2.3 Bounding box KVP encoding.
    
    Args:
        bbox: A string with four coodinates, optionally followed by a CRS spec.
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


def from_extent(extent: gws.Extent, crs: gws.ICrs, always_xy=False) -> gws.Bounds:
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
    return gws.Bounds(crs=b.crs, extent=b.extent)


def union(bs: list[gws.Bounds]) -> gws.Bounds:
    crs = bs[0].crs
    exts = [gws.gis.extent.transform(b.extent, b.crs, crs) for b in bs]
    return gws.Bounds(
        crs=crs,
        extent=gws.gis.extent.union(exts))


def transform(b: gws.Bounds, crs_to: gws.ICrs) -> gws.Bounds:
    if b.crs == crs_to:
        return b
    return gws.Bounds(
        crs=crs_to,
        extent=b.crs.transform_extent(b.extent, crs_to))
