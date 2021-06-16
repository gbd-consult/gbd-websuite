import gws

import gws.gis.proj
import gws.gis.gml
import gws.gis.extent
import gws.lib.xml2

import gws.types as t


#:export
class Bounds(t.Data):
    crs: t.Crs
    extent: t.Extent


def from_request_bbox(s: str, target_crs: t.Crs = None, invert_axis_if_geographic: bool = False) -> t.Optional[t.Bounds]:
    """Create Bounds from a KVP BBOX param.

    See:
        OGC 06-121r9, 10.2.3 Bounding box KVP encoding
    """

    crs = None
    s = s.split(',')
    if len(s) == 5:
        crs = s.pop()

    try:
        ext = [float(n) for n in s]
    except:
        return None

    proj = gws.gis.proj.as_proj(crs or target_crs)
    ext = gws.gis.extent.valid(ext)

    if ext and proj and proj.is_geographic and invert_axis_if_geographic:
        ext = gws.gis.extent.swap_xy(ext)

    if ext and proj:
        b = t.Bounds(crs=proj.epsg, extent=ext)
        if not gws.gis.proj.equal(b.crs, target_crs):
            b = transformed_to(b, target_crs)
        return b


def from_gml_envelope_element(el: gws.lib.xml2.Element):
    """Create Bounds from a gml:Envelope"""

    return gws.gis.gml.parse_envelope(el)


def transformed_to(b: t.Bounds, crs: t.Crs) -> t.Bounds:
    if gws.gis.proj.equal(b.crs, crs):
        return b
    proj = gws.gis.proj.as_proj(crs)
    return t.Bounds(
        crs=proj.epsg,
        extent=gws.gis.extent.transform(b.extent, b.crs, proj))
