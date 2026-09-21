"""GML geometry parsers."""

import gws
import gws.base.shape
import gws.lib.bounds
import gws.lib.crs
import gws.lib.extent


class Error(gws.Error):
    pass


_GEOMETRY_TAGS = [
    'Curve',
    'LinearRing',
    'LineString',
    'LineStringSegment',
    'MultiCurve',
    'MultiLineString',
    'MultiPoint',
    'MultiPolygon',
    'MultiSurface',
    'Point',
    'Polygon',
]


def parse_envelope(el: gws.XmlElement, default_crs: gws.Crs = None, always_xy: bool = False) -> gws.Bounds:
    """Parse a gml:Box/gml:Envelope element

    Args:
        el: A xml-Element.
        default_crs: A Crs object.
        always_xy: If ``True``, coordinates are assumed to be in the XY (lon/lat) order.

    Returns:
          A Bounds object.
    """

    # GML2: <gml:Box><gml:coordinates>1,2 3,4
    # GML3: <gml:Envelope srsDimension="2"><gml:lowerCorner>1 2  <gml:upperCorner>3 4

    crs = gws.lib.crs.get(el.get('srsName')) or default_crs
    if not crs:
        raise Error('no CRS declared for envelope')

    try:
        ext = _parse_envelope_extent(el)
    except Exception as exc:
        raise Error('envelope parse error') from exc

    return gws.lib.bounds.from_extent(ext, crs, always_xy)


def _parse_envelope_extent(el: gws.XmlElement) -> gws.Extent:
    if el.isa('Box'):
        a, b = _coords(el)
        return gws.lib.extent.from_points(a, b)

    if el.isa('Envelope'):
        a = b = None
        for coord_el in el:
            if coord_el.isa('lowerCorner'):
                a = _coords_pos(coord_el)[0]
            if coord_el.isa('upperCorner'):
                b = _coords_pos(coord_el)[0]
        if a and b:
            return gws.lib.extent.from_points(a, b)

    raise ValueError('invalid envelope element')


def is_geometry_element(el: gws.XmlElement) -> bool:
    """Checks if the current element is a valid geometry type.

    Args:
        el: A GML element.

    Returns:
        ``True`` if the element is a geometry type.
    """

    return el.isa(*_GEOMETRY_TAGS)


def parse_shape(el: gws.XmlElement, default_crs: gws.Crs = None, always_xy: bool = False) -> gws.Shape:
    """Convert a GML geometry element to a Shape.

    Args:
        el: A GML element.
        default_crs: A Crs object.
        always_xy: If ``True``, coordinates are assumed to be in the XY (lon/lat) order.

    Returns:
        A GWS shape object.
    """

    crs = gws.lib.crs.get(el.get('srsName')) or default_crs
    if not crs:
        raise Error('no CRS declared')

    dct = parse_geometry(el)
    return gws.base.shape.from_geojson(dct, crs, always_xy)


def parse_geometry(el: gws.XmlElement) -> dict:
    """Convert a GML geometry element to a geometry dict.

    Args:
        el: A GML element.

    Returns:
        The GML geometry as a geometry dict.
    """

    try:
        return _to_geom(el)
    except Exception as exc:
        raise Error('parse error') from exc


##


def _to_geom(el: gws.XmlElement):
    if el.isa('Point'):
        # <gml:Point> pos/coordinates
        return {'type': 'Point', 'coordinates': _coords(el)[0]}

    if el.isa('LineString', 'LinearRing', 'LineStringSegment'):
        # <gml:LineString> posList/coordinates
        return {'type': 'LineString', 'coordinates': _coords(el)}

    if el.isa('Curve'):
        # GML3: <gml:Curve> <gml:segments> <gml:LineStringSegment>
        # NB we only take the first segment
        return _to_geom(el[0][0])

    if el.isa('Polygon'):
        # GML2: <gml:Polygon> <gml:outerBoundaryIs> <gml:LinearRing> <gml:innerBoundaryIs> <gml:LinearRing>...
        # GML3: <gml:Polygon> <gml:exterior> <gml:LinearRing> <gml:interior> <gml:LinearRing>...
        return {'type': 'Polygon', 'coordinates': _rings(el)}

    if el.isa('MultiPoint'):
        # <gml:MultiPoint> <gml:pointMember> <gml:Point>
        return {'type': 'MultiPoint', 'coordinates': [m['coordinates'] for m in _members(el)]}

    if el.isa('MultiLineString', 'MultiCurve'):
        # GML2: <gml:MultiLineString> <gml:lineStringMember> <gml:LineString>
        # GML3: <gml:MultiCurve> <gml:curveMember> <gml:Curve>
        return {'type': 'MultiLineString', 'coordinates': [m['coordinates'] for m in _members(el)]}

    if el.isa('MultiPolygon', 'MultiSurface'):
        # GML2: <gml:MultiPolygon> <gml:polygonMember> <gml:Polygon>
        # GML3: <gml:MultiSurface> <gml:surfaceMember> <gml:Polygon>
        return {'type': 'MultiPolygon', 'coordinates': [m['coordinates'] for m in _members(el)]}

    raise Error(f'unknown GML geometry tag {el.name!r}')


def _members(multi_el: gws.XmlElement):
    ms = []

    for el in multi_el:
        if el.name.lower().endswith('member'):
            ms.append(_to_geom(el[0]))

    return ms


def _rings(poly_el):
    rings = [None]

    for el in poly_el:
        if el.isa('exterior', 'outerBoundaryIs'):
            d = _to_geom(el[0])
            rings[0] = d['coordinates']
            continue

        if el.isa('interior', 'innerBoundaryIs'):
            d = _to_geom(el[0])
            rings.append(d['coordinates'])
            continue

    return rings


def _coords(any_el):
    for el in any_el:
        if el.isa('coordinates'):
            return _coords_coordinates(el)
        if el.isa('pos'):
            return _coords_pos(el)
        if el.isa('posList'):
            return _coords_poslist(el)
    raise Error(f'expected coordinates list')


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
    return [(float(x), float(y))]


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
