"""Utilities to work with Bounds objects."""

import gws
import gws.lib.crs
import gws.lib.extent
import gws.lib.gml
import gws.lib.xml2
import gws.types as t


def from_request_bbox(bbox: str, target_crs: gws.ICrs = None, invert_axis_if_geographic: bool = False) -> t.Optional[gws.Bounds]:
    """Create Bounds from a KVP BBOX param.

    See:
        OGC 06-121r9, 10.2.3 Bounding box KVP encoding
    """

    source_crs = target_crs

    # x,y,x,y,crs
    ls = bbox.split(',')
    if len(ls) == 5:
        source_crs = gws.lib.crs.get(ls.pop())

    if not source_crs:
        return None

    if source_crs.is_geographic and invert_axis_if_geographic:
        ext = gws.lib.extent.from_inverted_str_list(ls)
    else:
        ext = gws.lib.extent.from_str_list(ls)
    if not ext:
        return None

    if target_crs:
        return gws.Bounds(
            crs=target_crs,
            extent=source_crs.transform_extent(ext, target_crs))

    return gws.Bounds(crs=source_crs, extent=ext)


def from_gml_envelope_element(el: gws.lib.xml2.Element):
    """Create Bounds from a gml:Envelope"""

    return gws.lib.gml.parse_envelope(el)


def transformed_to(b: gws.Bounds, target_crs: gws.ICrs) -> gws.Bounds:
    if target_crs.same_as(b.crs):
        return b

    return gws.Bounds(
        crs=target_crs,
        extent=b.crs.transform_extent(b.extent, target_crs))
