"""GML geometry parsers."""

import gws
import gws.gis.crs
import gws.gis.extent
import gws.gis.feature
import gws.gis.shape
import gws.lib.xml3 as xml3
import gws.types as t


class Error(gws.Error):
    pass


def parse_envelope(el: gws.XmlElement, fallback_crs: gws.ICrs = None) -> gws.Bounds:
    """Parse a gml:Box/gml:Envelope element"""

    # GML2: <gml:Box><gml:coordinates>1,2 3,4
    # GML3: <gml:Envelope srsDimension="2"><gml:lowerCorner>1 2  <gml:upperCorner>3 4

    crs = gws.gis.crs.get(xml3.attr(el, 'srsName')) or fallback_crs
    if not crs:
        raise Error('envelope: no CRS')

    name = _pname(el)
    coords = None

    try:
        if name == 'box':
            coords = _coords(el)

        elif name == 'Envelope':
            coords = []
            for c in el.children:
                cname = _pname(c)
                if cname == 'lowercorner':
                    coords[0] = _coords_pos(c)[0]
                if cname == 'uppercorner':
                    coords[1] = _coords_pos(c)[0]

        return gws.Bounds(
            crs=crs,
            extent=gws.gis.extent.from_points(*coords))

    except Exception as exc:
        raise Error('envelope: parse error') from exc


_GEOMETRY_TAGS = {
    'curve',
    'linearring',
    'linestring',
    'linestringsegment',
    'multicurve',
    'multilinestring',
    'multipoint',
    'multipolygon',
    'multisurface',
    'point',
    'polygon',
}


def element_is_gml(el: t.Optional[gws.XmlElement]) -> bool:
    if not el:
        return False
    return _pname(el) in _GEOMETRY_TAGS


def parse_to_shape(el: gws.XmlElement, fallback_crs: gws.ICrs = None) -> gws.IShape:
    """Convert a GML geometry element to a Shape."""

    crs = gws.gis.crs.get(xml3.attr(el, 'srsName')) or fallback_crs
    if not crs:
        raise Error('shape: no CRS')

    try:
        geometry = _to_geom(el)
    except Exception as exc:
        raise Error('shape: parse error') from exc

    return gws.gis.shape.from_geometry(geometry, crs)


def parse_to_geometry(el: gws.XmlElement) -> dict:
    """Convert a GML geometry element to a geometry dict."""

    try:
        return _to_geom(el)
    except Exception as exc:
        raise Error('shape: parse error') from exc


##

def _to_geom(el: gws.XmlElement):
    name = _pname(el)

    if name == 'point':
        # <gml:Point> pos/coordinates
        return {'type': 'Point', 'coordinates': _coords(el)[0]}

    if name == 'linestring' or name == 'linearring' or name == 'linestringsegment':
        # <gml:LineString> posList/coordinates
        return {'type': 'LineString', 'coordinates': _coords(el)}

    if name == 'curve':
        # GML3: <gml:Curve> <gml:segments> <gml:LineStringSegment>
        # NB we only take the first segment
        return _to_geom(el.children[0].children[0])

    if name == 'polygon':
        # GML2: <gml:Polygon> <gml:outerBoundaryIs> <gml:LinearRing> <gml:innerBoundaryIs> <gml:LinearRing>...
        # GML3: <gml:Polygon> <gml:exterior> <gml:LinearRing> <gml:interior> <gml:LinearRing>...
        return {'type': 'Polygon', 'coordinates': _rings(el)}

    if name == 'multipoint':
        # <gml:MultiPoint> <gml:pointMember> <gml:Point>
        return {'type': 'MultiPoint', 'coordinates': [m['coordinates'] for m in _members(el)]}

    if name == 'multilinestring' or name == 'multicurve':
        # GML2: <gml:MultiLineString> <gml:lineStringMember> <gml:LineString>
        # GML3: <gml:MultiCurve> <gml:curveMember> <gml:Curve>
        return {'type': 'MultiLineString', 'coordinates': [m['coordinates'] for m in _members(el)]}

    if name == 'multipolygon' or name == 'multisurface':
        # GML2: <gml:MultiPolygon> <gml:polygonMember> <gml:Polygon>
        # GML3: <gml:MultiSurface> <gml:surfaceMember> <gml:Polygon>
        return {'type': 'MultiPolygon', 'coordinates': [m['coordinates'] for m in _members(el)]}

    raise ValueError(f'unknown GML geometry tag {el.name!r}')


def _members(multi_el: gws.XmlElement):
    ms = []

    for c in multi_el.children:
        if _pname(c).endswith('member'):
            ms.append(_to_geom(c.children[0]))

    return ms


def _rings(poly_el):
    rings = [None]

    for c in poly_el.children:
        cname = _pname(c)

        if cname == 'exterior' or cname == 'outerboundaryis':
            d = _to_geom(c.children[0])
            rings[0] = d['coordinates']
            continue

        if cname == 'interior' or cname == 'innerboundaryis':
            d = _to_geom(c.children[0])
            rings.append(d['coordinates'])
            continue

    return rings


def _coords(el):
    for c in el.children:
        cname = _pname(c)

        if cname == 'coordinates':
            return _coords_coordinates(c)
        if cname == 'pos':
            return _coords_pos(c)
        if cname == 'poslist':
            return _coords_poslist(c)


def _coords_coordinates(el):
    # <gml:coordinates>1,2 3,4...

    ts = xml3.attr(el, 'ts', default=' ')
    cs = xml3.attr(el, 'cs', default=',')

    clist = []

    for pair in el.text.split(ts):
        x, y = pair.split(cs)
        clist.append([float(x), float(y)])

    return clist


def _coords_pos(el):
    # <gml:pos srsDimension="2">1 2</gml:pos>

    s = el.text.split()
    x = s[0]
    y = s[1]
    # NB pos returns a list of points too!
    return [[float(x), float(y)]]


def _coords_poslist(el):
    # <gml:posList srsDimension="2">1 2 3...

    clist = []
    dim = int(xml3.attr(el, 'srsDimension', default='2'))
    s = el.text.split()

    for n in range(0, len(s), dim):
        x = s[n]
        y = s[n + 1]
        clist.append([float(x), float(y)])

    return clist


def _pname(el):
    return xml3.unqualify_name(el.name).lower()
