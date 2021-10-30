import fiona.transform
import shapely.geometry
import shapely.geos as geos
import shapely.ops
import shapely.wkb
import shapely.wkt

import gws
import gws.lib.proj
import gws.types as t

_DEFAULT_POINT_BUFFER_RESOLUTION = 6
_MIN_TOLERANCE_POLYGON = 0.01  # 1 cm for metric projections
_CIRCLE_RESOLUTION = 64


def from_wkt(s: str, crs=None) -> gws.IShape:
    if s.startswith('SRID='):
        # EWKT
        c = s.index(';')
        crs = s[5:c]
        s = s[c + 1:]
    if not crs:
        raise ValueError('missing crs')
    geom = geos.WKTReader(geos.lgeos).read(s)
    return Shape(geom, crs)


def from_wkb(s: bytes, crs=None) -> gws.IShape:
    return _from_wkb(geos.WKBReader(geos.lgeos).read(s), crs)


def from_wkb_hex(s: str, crs=None) -> gws.IShape:
    return _from_wkb(geos.WKBReader(geos.lgeos).read_hex(s), crs)


def _from_wkb(g, crs):
    srid = geos.lgeos.GEOSGetSRID(g._geom)
    if srid:
        crs = srid
        geos.lgeos.GEOSSetSRID(g._geom, 0)
    if not crs:
        raise gws.Error('missing crs for WKB')
    return Shape(g, crs)


def from_geometry(geometry, crs) -> gws.IShape:
    if geometry.get('type').upper() == 'CIRCLE':
        geom = shapely.geometry.Point(geometry.get('center'))
        geom = geom.buffer(
            geometry.get('radius'),
            resolution=_CIRCLE_RESOLUTION,
            cap_style=shapely.geometry.CAP_STYLE.round,
            join_style=shapely.geometry.JOIN_STYLE.round)
    else:
        geom = shapely.geometry.shape(geometry)
    return Shape(geom, crs)


def from_props(props: gws.Data) -> gws.IShape:
    return from_geometry(
        props.get('geometry'),
        props.get('crs'))


def from_extent(extent: gws.Extent, crs) -> gws.IShape:
    return Shape(shapely.geometry.box(*extent), crs)


def from_bounds(bounds: gws.Bounds) -> gws.IShape:
    return Shape(shapely.geometry.box(*bounds.extent), bounds.crs)


def from_xy(x, y, crs) -> gws.IShape:
    return Shape(shapely.geometry.Point(x, y), crs)


def union(shapes) -> gws.IShape:
    if not shapes:
        raise ValueError('empty union')

    shapes = list(shapes)

    if len(shapes) == 0:
        raise ValueError('empty union')

    if len(shapes) == 1:
        return shapes[0]

    crs = shapes[0].crs
    shapes = [s.transformed_to(crs) for s in shapes]
    geom = shapely.ops.unary_union([getattr(s, 'geom') for s in shapes])

    return Shape(geom, crs)


class Props(gws.Props):
    crs: str
    geometry: dict


class Shape(gws.Object, gws.IShape):
    def __init__(self, geom, crs):
        super().__init__()
        self.geom: shapely.geometry.base.BaseGeometry = geom  # type: ignore
        p = gws.lib.proj.to_proj(crs)
        self.crs = p.epsg
        self.srid = p.srid

    @property
    def geometry_type(self) -> gws.GeometryType:
        return self.geom.type.upper()

    def props_for(self, user):
        return Props(
            crs=self.crs,
            geometry=shapely.geometry.mapping(self.geom))

    @property
    def wkb(self) -> bytes:
        return geos.WKBWriter(geos.lgeos).write(self.geom)

    @property
    def ewkb(self) -> bytes:
        geos.lgeos.GEOSSetSRID(self.geom._geom, self.srid)
        s = geos.WKBWriter(geos.lgeos, include_srid=True).write(self.geom)
        geos.lgeos.GEOSSetSRID(self.geom._geom, 0)
        return s

    @property
    def wkb_hex(self) -> str:
        return self.wkb.hex()

    @property
    def ewkb_hex(self) -> str:
        return self.ewkb.hex()

    @property
    def wkt(self) -> str:
        return geos.WKTWriter(geos.lgeos).write(self.geom)

    @property
    def ewkt(self) -> str:
        return f'SRID={self.srid};' + self.wkt

    @property
    def bounds(self) -> gws.Bounds:
        return gws.Bounds(crs=self.crs, extent=self.geom.bounds)

    @property
    def extent(self) -> gws.Extent:
        return self.geom.bounds

    @property
    def x(self) -> float:
        return getattr(self.geom, 'x', None)

    @property
    def y(self) -> float:
        return getattr(self.geom, 'y', None)

    @property
    def area(self) -> float:
        return getattr(self.geom, 'area', 0)

    @property
    def centroid(self) -> gws.IShape:
        return Shape(self.geom.centroid, self.crs)

    def intersects(self, shape: gws.IShape) -> bool:
        return self.geom.intersects(t.cast(Shape, shape).geom)

    def tolerance_polygon(self, tolerance, resolution=None) -> gws.IShape:
        is_poly = self.geometry_type in (gws.GeometryType.polygon, gws.GeometryType.multipolygon)

        if not tolerance and is_poly:
            return self

        # we need a polygon even if tolerance = 0
        tolerance = tolerance or _MIN_TOLERANCE_POLYGON
        resolution = resolution or _DEFAULT_POINT_BUFFER_RESOLUTION

        if is_poly:
            cs = shapely.geometry.CAP_STYLE.flat
            js = shapely.geometry.JOIN_STYLE.mitre
        else:
            cs = shapely.geometry.CAP_STYLE.round
            js = shapely.geometry.JOIN_STYLE.round

        geom = self.geom.buffer(tolerance, resolution, cap_style=cs, join_style=js)
        return Shape(geom, self.crs)

    def to_type(self, new_type: gws.GeometryType) -> gws.IShape:
        gt = self.geometry_type
        if new_type == gt:
            return self
        if new_type == gws.GeometryType.geometry:
            return self
        if gt == gws.GeometryType.point and new_type == gws.GeometryType.multipoint:
            return self.to_multi()
        if gt == gws.GeometryType.linestring and new_type == gws.GeometryType.multilinestring:
            return self.to_multi()
        if gt == gws.GeometryType.polygon and new_type == gws.GeometryType.multipolygon:
            return self.to_multi()
        raise ValueError(f'cannot convert {gt!r} to {new_type!r}')

    def to_multi(self) -> gws.IShape:
        gt = self.geometry_type
        if gt == gws.GeometryType.point:
            return Shape(shapely.geometry.MultiPoint([self.geom]), self.crs)
        if gt == gws.GeometryType.linestring:
            return Shape(shapely.geometry.MultiLineString([self.geom]), self.crs)
        if gt == gws.GeometryType.polygon:
            return Shape(shapely.geometry.MultiPolygon([self.geom]), self.crs)
        return self

    def to_geojson(self):
        # @TODO
        return "..."

    def transformed_to(self, to_crs, **kwargs) -> gws.IShape:
        if gws.lib.proj.equal(self.crs, to_crs):
            return self
        to_crs = gws.lib.proj.to_proj(to_crs).epsg
        sg = shapely.geometry.mapping(self.geom)
        dg = fiona.transform.transform_geom(self.crs, to_crs, sg, **kwargs)
        return Shape(shapely.geometry.shape(dg), to_crs)
