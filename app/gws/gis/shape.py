import shapely.geometry
import shapely.wkb
import shapely.wkt
import shapely.wkt
import shapely.ops

import gws
import gws.gis.proj
import gws.types as t


def from_wkt(wkt, crs) -> t.IShape:
    return Shape(shapely.wkt.loads(wkt), crs)


def from_wkb(wkb, crs, hex=True) -> t.IShape:
    return Shape(shapely.wkb.loads(wkb, hex), crs)


def from_geometry(geometry, crs) -> t.IShape:
    if geometry.get('type') == 'Circle':
        geom = shapely.geometry.Point(geometry.get('center'))
        geom = geom.buffer(geometry.get('radius'), 6)
    else:
        geom = shapely.geometry.shape(geometry)
    return Shape(geom, crs)


def from_props(props: t.ShapeProps) -> t.IShape:
    return from_geometry(
        props.get('geometry'),
        props.get('crs'))


def from_bbox(bbox, crs) -> t.IShape:
    return Shape(shapely.geometry.box(*bbox), crs)


def from_xy(x, y, crs) -> t.IShape:
    return Shape(shapely.geometry.Point(x, y), crs)


def union(shapes: t.List[t.IShape]) -> t.IShape:
    if not shapes:
        return []

    shapes = list(shapes)
    if len(shapes) == 0:
        return []
    if len(shapes) == 1:
        return shapes[0]

    crs = shapes[0].crs
    shapes = [s.transform(crs) for s in shapes]
    geom = shapely.ops.unary_union([getattr(s, 'geom') for s in shapes])

    return Shape(geom, crs)


_DEFAULT_POINT_BUFFER_RESOLUTION = 6


def buffer_point(sh, tolerance, resolution=_DEFAULT_POINT_BUFFER_RESOLUTION) -> t.Optional[t.IShape]:
    if sh.type != 'Point':
        return
    if not tolerance:
        return
    return Shape(sh.geom.buffer(tolerance, resolution), sh.crs)


#:export
class ShapeProps(t.Props):
    geometry: dict
    crs: str


#:export IShape
class Shape(t.IShape):
    def __init__(self, geom, crs):
        self.crs: t.Crs = gws.gis.proj.as_epsg(crs)
        #:noexport
        self.geom: shapely.geometry.base.BaseGeometry = geom

    @property
    def type(self) -> str:
        return self.geom.type

    @property
    def props(self) -> t.ShapeProps:
        return t.ShapeProps({
            'crs': self.crs,
            'geometry': shapely.geometry.mapping(self.geom),
        })

    @property
    def wkb(self) -> str:
        return self.geom.wkb

    @property
    def wkb_hex(self) -> str:
        return self.geom.wkb_hex

    @property
    def wkt(self) -> str:
        return self.geom.wkt

    @property
    def bounds(self) -> t.Extent:
        return self.geom.bounds

    @property
    def x(self) -> float:
        return getattr(self.geom, 'x', None)

    @property
    def y(self) -> float:
        return getattr(self.geom, 'y', None)

    @property
    def centroid(self) -> t.IShape:
        return Shape(self.geom.centroid, self.crs)

    def intersects(self, shape: t.IShape) -> bool:
        s: Shape = shape
        return self.geom.intersects(s.geom)

    def tolerance_buffer(self, tolerance, resolution=None) -> t.IShape:
        if not tolerance:
            return self
        return Shape(self.geom.buffer(tolerance, resolution or _DEFAULT_POINT_BUFFER_RESOLUTION), self.crs)

    def transform(self, to_crs) -> t.IShape:
        if gws.gis.proj.equal(to_crs, self.crs):
            return self
        return Shape(gws.gis.proj.transform(self.geom, self.crs, to_crs), to_crs)
