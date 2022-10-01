"""Utilities to work with Bounds objects."""

import gws
import gws.gis.crs
import gws.gis.extent
import gws.gis.gml
import gws.types as t


def from_request_bbox(bbox: str, target_crs: gws.ICrs = None, invert_axis_if_geographic: bool = False) -> t.Optional[gws.Bounds]:
    """Create Bounds from a KVP BBOX param.

    See:
        OGC 06-121r9, 10.2.3 Bounding box KVP encoding
    """

    if not bbox:
        return None

    source_crs = target_crs

    # x,y,x,y,crs
    ls = bbox.split(',')
    if len(ls) == 5:
        source_crs = gws.gis.crs.get(ls.pop())

    if not source_crs:
        return None

    if source_crs.is_geographic and invert_axis_if_geographic:
        ext = gws.gis.extent.from_inverted_str_list(ls)
    else:
        ext = gws.gis.extent.from_list(ls)
    if not ext:
        return None

    if target_crs:
        return gws.Bounds(
            crs=target_crs,
            extent=source_crs.transform_extent(ext, target_crs))

    return gws.Bounds(crs=source_crs, extent=ext)


def from_gml_envelope_element(el: gws.IXmlElement, fallback_crs: gws.ICrs = None):
    """Create Bounds from a gml:Envelope"""

    return gws.gis.gml.parse_envelope(el, fallback_crs)


def copy(b: gws.Bounds) -> gws.Bounds:
    return gws.Bounds(crs=b.crs, extent=b.extent)


def union(bs: t.List[gws.Bounds]) -> gws.Bounds:
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



