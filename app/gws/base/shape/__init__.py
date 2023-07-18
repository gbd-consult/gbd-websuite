"""Shape object.

The Shape object implements the IShape protocol (georefenced geometry).
Internally, it holds a pointer to a Shapely geometry object and a Crs object.
"""

# @TODO support for SQL/MM extensions

import struct
import shapely.geometry
import shapely.ops
import shapely.wkb
import shapely.wkt

import gws
import gws.gis.crs
import gws.lib.sa as sa

_TOLERANCE_QUAD_SEGS = 6
_MIN_TOLERANCE_POLYGON = 0.01  # 1 cm for metric projections


class Error(gws.Error):
    pass


def from_wkt(wkt: str, default_crs: gws.ICrs = None) -> gws.IShape:
    """Creates a shape object from a WKT string.

    Args:
        wkt: A WKT or EWKT string.
        default_crs: Default Crs.

    Returns:
        A Shape object.
    """

    if wkt.startswith('SRID='):
        # EWKT
        c = wkt.index(';')
        srid = wkt[len('SRID='):c]
        crs = gws.gis.crs.get(int(srid))
        wkt = wkt[c + 1:]
    elif default_crs:
        crs = default_crs
    else:
        raise Error('missing or invalid crs for WKT')

    return Shape(shapely.wkt.loads(wkt), crs)


def from_wkb(wkb: bytes, default_crs: gws.ICrs = None) -> gws.IShape:
    """Creates a shape object from a WKB byte string.

    Args:
        wkb: A WKB or EWKB byte string.
        default_crs: Default Crs.

    Returns:
        A Shape object.
    """

    return _from_wkb(wkb, default_crs)


def from_wkb_hex(wkb: str, default_crs: gws.ICrs = None) -> gws.IShape:
    """Creates a shape object from a hex-encoded WKB string.

    Args:
        wkb: A hex-encoded WKB or EWKB byte string.
        default_crs: Default Crs.

    Returns:
        A Shape object.
    """

    return _from_wkb(bytes.fromhex(wkb), default_crs)


def _from_wkb(wkb: bytes, default_crs):
    # http://libgeos.org/specifications/wkb/#extended-wkb

    byte_order = wkb[0]
    header = struct.unpack('<cLL' if byte_order == 1 else '>cLL', wkb[:9])

    if header[1] & 0x20000000:
        crs = gws.gis.crs.get(header[2])
    elif default_crs:
        crs = default_crs
    else:
        raise Error('missing or invalid crs for WKB')

    geom = shapely.wkb.loads(wkb)
    return Shape(geom, crs)


def from_wkb_element(element: sa.geo.WKBElement, default_crs):
    data = element.data
    if isinstance(data, str):
        wkb = bytes.fromhex(data)
    else:
        wkb = bytes(data)
    crs = gws.gis.crs.get(element.srid)
    return _from_wkb(wkb, crs or default_crs)


def from_geojson(geojson: dict, crs: gws.ICrs, always_xy=False) -> gws.IShape:
    """Creates a shape object from a GeoJSON geometry dict.

    Parses a dict as a GeoJSON geometry object (https://www.rfc-editor.org/rfc/rfc7946#section-3.1).

    The coordinates are assumed to be in the projection order, unless ``always_xy`` is ``True``.

    Args:
        geojson: A GeoJSON geometry dict
        crs: A Crs object.
        always_xy: If ``True``, coordinates are assumed to be in the XY (lon/lat) order

    Returns:
        A Shape object.
    """

    geom = _shapely_shape(geojson)
    if crs.isYX and not always_xy:
        geom = _swap_xy(geom)
    return Shape(geom, crs)


def from_props(props: gws.Props) -> gws.IShape:
    """Creates a Shape from a properties object.

    Args:
        props: A properties object.
    Returns:
        A Shape object.
    """

    crs = gws.gis.crs.get(props.get('crs'))
    if not crs:
        raise Error('missing or invalid crs')
    geom = _shapely_shape(props.get('geometry'))
    return Shape(geom, crs)


def from_dict(d: dict) -> gws.IShape:
    """Creates a Shape from a dictionary.

    Args:
        d: A dictionary with the keys 'crs' and 'geometry'.
    Returns:
        A Shape object.
    """

    crs = gws.gis.crs.get(d.get('crs'))
    if not crs:
        raise Error('missing or invalid crs')
    geom = _shapely_shape(d.get('geometry'))
    return Shape(geom, crs)


def from_extent(extent: gws.Extent, crs: gws.ICrs, always_xy=False) -> gws.IShape:
    """Creates a polygon Shape from an extent.

    Args:
        extent: A hex-encoded WKB byte string.
        crs: A Crs object.
        always_xy: If ``True``, coordinates are assumed to be in the XY (lon/lat) order

    Returns:
        A Shape object.
    """

    geom = shapely.geometry.box(*extent)
    if crs.isYX and not always_xy:
        geom = _swap_xy(geom)
    return Shape(geom, crs)


def from_bounds(bounds: gws.Bounds) -> gws.IShape:
    """Creates a polygon Shape from a Bounds object.

    Args:
        bounds: A Bounds object.

    Returns:
        A Shape object.
    """

    return Shape(shapely.geometry.box(*bounds.extent), bounds.crs)


def from_xy(x: float, y: float, crs: gws.ICrs) -> gws.IShape:
    """Creates a point Shape from coordinates.

    Args:
        x: X coordinate (lon/easting)
        y: Y coordinate (lat/northing)
        crs: A Crs object.

    Returns:
        A Shape object.
    """

    return Shape(shapely.geometry.Point(x, y), crs)


def _swap_xy(geom):
    def f(x, y):
        return y, x

    return shapely.ops.transform(f, geom)


_CIRCLE_RESOLUTION = 64


def _shapely_shape(d):
    if d.get('type').upper() == 'CIRCLE':
        geom = shapely.geometry.Point(d.get('center'))
        return geom.buffer(
            d.get('radius'),
            resolution=_CIRCLE_RESOLUTION,
            cap_style=shapely.geometry.CAP_STYLE.round,
            join_style=shapely.geometry.JOIN_STYLE.round)

    return shapely.geometry.shape(d)


##


class Props(gws.Props):
    """Shape properties object."""
    crs: str
    geometry: dict


##


class Shape(gws.Object, gws.IShape):
    geom: shapely.geometry.base.BaseGeometry

    def __init__(self, geom, crs: gws.ICrs):
        self.geom = geom
        self.crs = crs
        self.type = self.geom.geom_type.lower()
        self.x = getattr(self.geom, 'x', None)
        self.y = getattr(self.geom, 'y', None)

    def props(self, user):
        return Props(
            crs=self.crs.epsg,
            geometry=shapely.geometry.mapping(self.geom))

    def area(self):
        return getattr(self.geom, 'area', 0)

    def bounds(self):
        return gws.Bounds(crs=self.crs, extent=self.geom.bounds)

    def centroid(self):
        return Shape(self.geom.centroid, self.crs)

    def to_wkb(self):
        return shapely.wkb.dumps(self.geom)

    def to_wkb_hex(self):
        return shapely.wkb.dumps(self.geom, hex=True)

    def to_ewkb(self):
        return shapely.wkb.dumps(self.geom, srid=self.crs.srid)

    def to_ewkb_hex(self):
        return shapely.wkb.dumps(self.geom, srid=self.crs.srid, hex=True)

    def to_wkt(self):
        return shapely.wkt.dumps(self.geom)

    def to_ewkt(self):
        return f'SRID={self.crs.srid};' + self.to_wkt()

    def to_geojson(self, always_xy=False):
        geom = self.geom
        if self.crs.isYX and not always_xy:
            geom = _swap_xy(geom)
        return shapely.geometry.mapping(geom)

    def is_empty(self):
        return self.geom.is_empty()

    def is_ring(self):
        return self.geom.is_ring()

    def is_simple(self):
        return self.geom.is_simple()

    def is_valid(self):
        return self.geom.is_valid()

    def equals(self, other):
        return self._binary_predicate(other, 'equals')

    def contains(self, other):
        return self._binary_predicate(other, 'contains')

    def covers(self, other):
        return self._binary_predicate(other, 'covers')

    def covered_by(self, other):
        return self._binary_predicate(other, 'covered_by')

    def crosses(self, other):
        return self._binary_predicate(other, 'crosses')

    def disjoint(self, other):
        return self._binary_predicate(other, 'disjoint')

    def intersects(self, other):
        return self._binary_predicate(other, 'intersects')

    def overlaps(self, other):
        return self._binary_predicate(other, 'overlaps')

    def touches(self, other):
        return self._binary_predicate(other, 'touches')

    def within(self, other):
        return self._binary_predicate(other, 'within')

    def _binary_predicate(self, other, op):
        s = other.transformed_to(self.crs)
        return getattr(self.geom, op)(getattr(s, 'geom'))

    def union(self, others):
        if not others:
            return self

        geoms = [self.geom]
        for s in others:
            s = s.transformed_to(self.crs)
            geoms.append(getattr(s, 'geom'))

        geom = shapely.ops.unary_union(geoms)
        return Shape(geom, self.crs)

    def intersection(self, *others):
        if not others:
            return self

        geom = self.geom
        for s in others:
            s = s.transformed_to(self.crs)
            geom = geom.intersection(getattr(s, 'geom'))

        return Shape(geom, self.crs)

    def to_multi(self):
        if self.type == gws.GeometryType.point:
            return Shape(shapely.geometry.MultiPoint([self.geom]), self.crs)
        if self.type == gws.GeometryType.linestring:
            return Shape(shapely.geometry.MultiLineString([self.geom]), self.crs)
        if self.type == gws.GeometryType.polygon:
            return Shape(shapely.geometry.MultiPolygon([self.geom]), self.crs)
        return self

    def to_type(self, new_type: gws.GeometryType):
        if new_type == self.type:
            return self
        if new_type == gws.GeometryType.geometry:
            return self
        if self.type == gws.GeometryType.point and new_type == gws.GeometryType.multipoint:
            return self.to_multi()
        if self.type == gws.GeometryType.linestring and new_type == gws.GeometryType.multilinestring:
            return self.to_multi()
        if self.type == gws.GeometryType.polygon and new_type == gws.GeometryType.multipolygon:
            return self.to_multi()
        raise Error(f'cannot convert {self.type!r} to {new_type!r}')

    def tolerance_polygon(self, tolerance, quad_segs=None):
        is_poly = self.type in (gws.GeometryType.polygon, gws.GeometryType.multipolygon)

        if not tolerance and is_poly:
            return self

        # we need a polygon even if tolerance = 0
        tolerance = tolerance or _MIN_TOLERANCE_POLYGON
        quad_segs = quad_segs or _TOLERANCE_QUAD_SEGS

        if is_poly:
            cs = shapely.geometry.CAP_STYLE.flat
            js = shapely.geometry.JOIN_STYLE.mitre
        else:
            cs = shapely.geometry.CAP_STYLE.round
            js = shapely.geometry.JOIN_STYLE.round

        geom = self.geom.buffer(tolerance, quad_segs, cap_style=cs, join_style=js)
        return Shape(geom, self.crs)

    def transformed_to(self, crs):
        if crs == self.crs:
            return self
        tr = self.crs.transformer(crs)
        dg = shapely.ops.transform(tr, self.geom)
        return Shape(dg, crs)
