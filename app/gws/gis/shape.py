import shapely.geometry
import shapely.wkb
import shapely.wkt
import shapely.wkt
import shapely.ops
import shapely.geos as geos

import fiona.transform

import gws
import gws.gis.proj
import gws.types as t

_DEFAULT_POINT_BUFFER_RESOLUTION = 6
_CIRCLE_RESOLUTION = 16


def from_wkt(s: str, crs=None) -> t.IShape:
    if s.startswith('SRID='):
        # EWKT
        c = s.index(';')
        crs = s[5:c]
        s = s[c + 1:]
    if not crs:
        raise ValueError('missing crs')
    geom = geos.WKTReader(geos.lgeos).read(s)
    return Shape(geom, crs)


def from_wkb(s: bytes, crs=None) -> t.IShape:
    return _from_wkb(geos.WKBReader(geos.lgeos).read(s), crs)


def from_wkb_hex(s: str, crs=None) -> t.IShape:
    return _from_wkb(geos.WKBReader(geos.lgeos).read_hex(s), crs)


def _from_wkb(g, crs):
    srid = geos.lgeos.GEOSGetSRID(g._geom)
    if srid:
        crs = srid
        geos.lgeos.GEOSSetSRID(g._geom, 0)
    if not crs:
        raise ValueError('missing crs')
    return Shape(g, crs)


def from_geometry(geometry, crs) -> t.IShape:
    if geometry.get('type').lower() == 'circle':
        geom = shapely.geometry.Point(geometry.get('center'))
        geom = geom.buffer(
            geometry.get('radius'),
            resolution=_CIRCLE_RESOLUTION,
            cap_style=shapely.geometry.CAP_STYLE.round,
            join_style=shapely.geometry.JOIN_STYLE.round)
    else:
        geom = shapely.geometry.shape(geometry)
    return Shape(geom, crs)


def from_props(props: t.ShapeProps) -> t.IShape:
    return from_geometry(
        props.get('geometry'),
        props.get('crs'))


def from_extent(extent: t.Extent, crs) -> t.IShape:
    return Shape(shapely.geometry.box(*extent), crs)


def from_bounds(bounds: t.Bounds) -> t.IShape:
    return Shape(shapely.geometry.box(*bounds.extent), bounds.crs)


def from_xy(x, y, crs) -> t.IShape:
    return Shape(shapely.geometry.Point(x, y), crs)


def union(shapes) -> t.Optional[t.IShape]:
    if not shapes:
        return

    shapes = list(shapes)
    if len(shapes) == 0:
        return
    if len(shapes) == 1:
        return shapes[0]

    crs = shapes[0].crs
    shapes = [s.transformed(crs) for s in shapes]
    geom = shapely.ops.unary_union([getattr(s, 'geom') for s in shapes])

    return Shape(geom, crs)


#:export
class ShapeProps(t.Props):
    crs: str
    geometry: dict


#:export IShape
class Shape(t.IShape):
    def __init__(self, geom, crs):
        p = gws.gis.proj.as_proj(crs)
        self.crs: str = p.epsg
        self.srid: int = p.srid
        #:noexport
        self.geom: shapely.geometry.base.BaseGeometry = geom

    @property
    def type(self) -> t.GeometryType:
        return self.geom.type.lower()

    @property
    def props(self) -> t.ShapeProps:
        return t.ShapeProps(
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
    def bounds(self) -> t.Bounds:
        return t.Bounds(crs=self.crs, extent=self.geom.bounds)

    @property
    def extent(self) -> t.Extent:
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
    def centroid(self) -> t.IShape:
        return Shape(self.geom.centroid, self.crs)

    def intersects(self, shape: t.IShape) -> bool:
        s: Shape = shape
        return self.geom.intersects(s.geom)

    def tolerance_buffer(self, tolerance, resolution=None) -> t.IShape:
        if not tolerance:
            return self

        resolution = resolution or _DEFAULT_POINT_BUFFER_RESOLUTION

        if 'polygon' in self.type:
            cs = shapely.geometry.CAP_STYLE.flat
            js = shapely.geometry.JOIN_STYLE.mitre
        else:
            cs = shapely.geometry.CAP_STYLE.round
            js = shapely.geometry.JOIN_STYLE.round

        geom = self.geom.buffer(tolerance, resolution, cap_style=cs, join_style=js)
        return Shape(geom, self.crs)

    def transformed(self, to_crs, **kwargs) -> t.IShape:
        if gws.gis.proj.equal(self.crs, to_crs):
            return self
        to_crs = gws.gis.proj.as_proj(to_crs).epsg
        sg = shapely.geometry.mapping(self.geom)
        dg = fiona.transform.transform_geom(self.crs, to_crs, sg, **kwargs)
        return Shape(shapely.geometry.shape(dg), to_crs)
