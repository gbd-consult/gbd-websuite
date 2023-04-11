"""GML geometry parsers."""

import gws
import gws.base.shape
import gws.gis.bounds
import gws.gis.crs
import gws.gis.extent


class Error(gws.Error):
    pass


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


def parse_envelope(el: gws.IXmlElement, default_crs: gws.ICrs = None, always_xy=False) -> gws.Bounds:
    """Parse a gml:Box/gml:Envelope element"""

    # GML2: <gml:Box><gml:coordinates>1,2 3,4
    # GML3: <gml:Envelope srsDimension="2"><gml:lowerCorner>1 2  <gml:upperCorner>3 4

    crs = gws.gis.crs.get(el.get('srsName')) or default_crs
    if not crs:
        raise Error('no CRS declared for envelope')

    coords = None

    try:
        if el.lname == 'box':
            coords = _coords(el)

        elif el.lname == 'envelope':
            coords = []
            for coord_el in el:
                if coord_el.lname == 'lowercorner':
                    coords[0] = _coords_pos(coord_el)[0]
                if coord_el.lname == 'uppercorner':
                    coords[1] = _coords_pos(coord_el)[0]

        ext = gws.gis.extent.from_points(*coords)

    except Exception as exc:
        raise Error('envelope parse error') from exc

    return gws.gis.bounds.from_extent(ext, crs, always_xy)


def is_geometry_element(el: gws.IXmlElement):
    return el.lname in _GEOMETRY_TAGS


def parse_shape(el: gws.IXmlElement, default_crs: gws.ICrs = None, always_xy=False) -> gws.IShape:
    """Convert a GML geometry element to a Shape."""

    crs = gws.gis.crs.get(el.get('srsName')) or default_crs
    if not crs:
        raise Error('no CRS declared')

    dct = parse_geometry(el)
    return gws.base.shape.from_geojson(dct, crs, always_xy)


def parse_geometry(el: gws.IXmlElement) -> dict:
    """Convert a GML geometry element to a geometry dict."""

    try:
        return _to_geom(el)
    except Exception as exc:
        raise Error('parse error') from exc


##

def _to_geom(el: gws.IXmlElement):
    if el.lname == 'point':
        # <gml:Point> pos/coordinates
        return {'type': 'Point', 'coordinates': _coords(el)[0]}

    if el.lname in {'linestring', 'linearring', 'linestringsegment'}:
        # <gml:LineString> posList/coordinates
        return {'type': 'LineString', 'coordinates': _coords(el)}

    if el.lname == 'curve':
        # GML3: <gml:Curve> <gml:segments> <gml:LineStringSegment>
        # NB we only take the first segment
        return _to_geom(el[0][0])

    if el.lname == 'polygon':
        # GML2: <gml:Polygon> <gml:outerBoundaryIs> <gml:LinearRing> <gml:innerBoundaryIs> <gml:LinearRing>...
        # GML3: <gml:Polygon> <gml:exterior> <gml:LinearRing> <gml:interior> <gml:LinearRing>...
        return {'type': 'Polygon', 'coordinates': _rings(el)}

    if el.lname == 'multipoint':
        # <gml:MultiPoint> <gml:pointMember> <gml:Point>
        return {'type': 'MultiPoint', 'coordinates': [m['coordinates'] for m in _members(el)]}

    if el.lname in {'multilinestring', 'multicurve'}:
        # GML2: <gml:MultiLineString> <gml:lineStringMember> <gml:LineString>
        # GML3: <gml:MultiCurve> <gml:curveMember> <gml:Curve>
        return {'type': 'MultiLineString', 'coordinates': [m['coordinates'] for m in _members(el)]}

    if el.lname == {'multipolygon', 'multisurface'}:
        # GML2: <gml:MultiPolygon> <gml:polygonMember> <gml:Polygon>
        # GML3: <gml:MultiSurface> <gml:surfaceMember> <gml:Polygon>
        return {'type': 'MultiPolygon', 'coordinates': [m['coordinates'] for m in _members(el)]}

    raise Error(f'unknown GML geometry tag {el.name!r}')


def _members(multi_el: gws.IXmlElement):
    ms = []

    for el in multi_el:
        if el.lname.endswith('member'):
            ms.append(_to_geom(el[0]))

    return ms


def _rings(poly_el):
    rings = [None]

    for el in poly_el:
        if el.lname in {'exterior', 'outerboundaryis'}:
            d = _to_geom(el[0])
            rings[0] = d['coordinates']
            continue

        if el.lname in {'interior', 'innerboundaryis'}:
            d = _to_geom(el[0])
            rings.append(d['coordinates'])
            continue

    return rings


def _coords(any_el):
    for el in any_el:
        if el.lname == 'coordinates':
            return _coords_coordinates(el)
        if el.lname == 'pos':
            return _coords_pos(el)
        if el.lname == 'poslist':
            return _coords_poslist(el)


def _coords_coordinates(el):
    # <gml:coordinates>1,2 3,4...

    ts = el.get('ts', default=' ')
    cs = el.get('cs', default=',')

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
    dim = int(el.get('srsDimension', default='2'))
    s = el.text.split()

    for n in range(0, len(s), dim):
        x = s[n]
        y = s[n + 1]
        clist.append([float(x), float(y)])

    return clist
