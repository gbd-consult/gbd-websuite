import gws
import gws.types as t
import gws
import gws.types as t
import gws.lib.extent
import gws.lib.gml
import gws.lib.proj
import gws.lib.xml2


def from_request_bbox(s: str, target_crs: gws.Crs = None, invert_axis_if_geographic: bool = False) -> t.Optional[gws.Bounds]:
    """Create Bounds from a KVP BBOX param.

    See:
        OGC 06-121r9, 10.2.3 Bounding box KVP encoding
    """

    crs = target_crs
    ls = s.split(',')
    if len(ls) == 5:
        crs = ls.pop()

    proj = gws.lib.proj.to_proj(crs)
    if not proj:
        return None

    if proj.is_geographic and invert_axis_if_geographic:
        ls = [ls[1], ls[0], ls[3], ls[2]]

    ext = gws.lib.extent.from_str_list(ls)
    if not ext:
        return None

    b = gws.Bounds(crs=proj.epsg, extent=ext)
    if target_crs and not gws.lib.proj.equal(b.crs, target_crs):
        b = transformed_to(b, target_crs)

    return b


def from_gml_envelope_element(el: gws.lib.xml2.Element):
    """Create Bounds from a gml:Envelope"""

    return gws.lib.gml.parse_envelope(el)


def transformed_to(b: gws.Bounds, crs: gws.Crs) -> gws.Bounds:
    if gws.lib.proj.equal(b.crs, crs):
        return b
    proj = gws.lib.proj.to_proj(crs)
    return gws.Bounds(
        crs=proj.epsg,
        extent=gws.lib.extent.transform(b.extent, b.crs, proj))
