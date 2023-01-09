"""Shape object.

The Shape object implements the IShape protocol (georefenced geometry).
Internally, it holds a pointer to a Shapely geometry object and a Crs object.
"""

# @TODO support for SQL/MM extensions

import shapely.geometry
import shapely.geos as geos
import shapely.ops
import shapely.wkb
import shapely.wkt

import gws
import gws.gis.crs


def from_wkt(wkt: str, default_crs: gws.ICrs = None) -> gws.IShape:
    """Creates a shape object from a WKT string.

    Args:
        wkt: A WKT or EWKT string.
        default_crs: Default Crs.

    Returns:
        A Shape object.
    """

    crs = default_crs

    if wkt.startswith('SRID='):
        # EWKT
        c = wkt.index(';')
        crsid = wkt[len('SRID='):c]
        wkt = wkt[c + 1:]
        crs = gws.gis.crs.get(crsid)

    if not crs:
        raise gws.Error('missing or invalid crs for WKT')

    geom = geos.WKTReader(geos.lgeos).read(wkt)
    return Shape(geom, crs)


def from_wkb(wkb: bytes, default_crs: gws.ICrs = None) -> gws.IShape:
    """Creates a shape object from a WKB byte string.

    Args:
        wkb: A WKB or EWKB byte string.
        default_crs: Default Crs.

    Returns:
        A Shape object.
    """

    return _from_wkb(geos.WKBReader(geos.lgeos).read(wkb), default_crs)


def from_wkb_hex(wkb: str, default_crs: gws.ICrs = None) -> gws.IShape:
    """Creates a shape object from a hex-encoded WKB byte string.

    Args:
        wkb: A hex-encoded WKB or EWKB byte string.
        default_crs: Default Crs.

    Returns:
        A Shape object.
    """

    return _from_wkb(geos.WKBReader(geos.lgeos).read_hex(wkb), default_crs)


def _from_wkb(g, default_crs):
    crs = default_crs

    crsid = geos.lgeos.GEOSGetSRID(g._geom)
    if crsid:
        crs = gws.gis.crs.get(crsid)
        geos.lgeos.GEOSSetSRID(g._geom, 0)

    if not crs:
        raise gws.Error('missing or invalid crs for WKT')

    return Shape(g, crs)


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

    # _CIRCLE_RESOLUTION = 64
    # if geojson.get('type').upper() == 'CIRCLE':
    #     geom = shapely.geometry.Point(geojson.get('center'))
    #     geom = geom.buffer(
    #         geojson.get('radius'),
    #         resolution=_CIRCLE_RESOLUTION,
    #         cap_style=shapely.geometry.CAP_STYLE.round,
    #         join_style=shapely.geometry.JOIN_STYLE.round)

    geom = shapely.geometry.shape(geojson)
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
        raise gws.Error('missing or invalid crs')
    return from_geojson(props.get('geometry'), crs)


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


##


class Props(gws.Props):
    """Shape properties object."""
    crs: str
    geometry: dict


##


class Shape(gws.Object, gws.IShape):
    geom: shapely.geometry.base.BaseGeometry

    def __init__(self, geom, crs):
        self.geom = geom
        self.crs = crs
        self.type = self.geom.type.lower()
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
        return geos.WKBWriter(geos.lgeos).write(self.geom)

    def to_wkb_hex(self):
        return self.to_wkb().hex()

    def to_ewkb(self):
        geos.lgeos.GEOSSetSRID(self.geom._geom, self.crs.srid)
        s = geos.WKBWriter(geos.lgeos, include_srid=True).write(self.geom)
        geos.lgeos.GEOSSetSRID(self.geom._geom, 0)
        return s

    def to_ewkb_hex(self):
        return self.to_ewkb().hex()

    def to_wkt(self):
        return geos.WKTWriter(geos.lgeos).write(self.geom)

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
        raise ValueError(f'cannot convert {self.type!r} to {new_type!r}')

    _TOLERANCE_QUAD_SEGS = 6
    _MIN_TOLERANCE_POLYGON = 0.01  # 1 cm for metric projections

    def tolerance_polygon(self, tolerance, quad_segs=None):
        is_poly = self.type in (gws.GeometryType.polygon, gws.GeometryType.multipolygon)

        if not tolerance and is_poly:
            return self

        # we need a polygon even if tolerance = 0
        tolerance = tolerance or self._MIN_TOLERANCE_POLYGON
        quad_segs = quad_segs or self._TOLERANCE_QUAD_SEGS

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
