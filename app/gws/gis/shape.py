import shapely.geometry
import shapely.wkb
import shapely.wkt
import shapely.wkt
import shapely.ops

import gws
import gws.gis.proj
import gws.types as t


def from_wkt(wkt, crs):
    return Shape(shapely.wkt.loads(wkt), crs)


def from_wkb(wkb, crs, hex=True):
    return Shape(shapely.wkb.loads(wkb, hex), crs)


def from_geometry(geometry, crs):
    if geometry.get('type') == 'Circle':
        geo = shapely.geometry.Point(geometry.get('center'))
        geo = geo.buffer(geometry.get('radius'), 6)
        return Shape(geo, crs)

    return Shape(
        shapely.geometry.shape(geometry),
        crs)


def from_props(props: t.ShapeProps):
    return from_geometry(
        props.get('geometry'),
        props.get('crs'))


def from_bbox(bbox, crs):
    return Shape(shapely.geometry.box(*bbox), crs)


def from_xy(x, y, crs):
    return Shape(shapely.geometry.Point(x, y), crs)


def merge_extents(exts):
    ex = [1e20, 1e20, 0, 0]
    for e in exts:
        minx = min(e[0], e[2])
        maxx = max(e[0], e[2])
        miny = min(e[1], e[3])
        maxy = max(e[1], e[3])
        ex = [
            min(ex[0], minx),
            min(ex[1], miny),
            max(ex[2], maxx),
            max(ex[3], maxy)
        ]
    return ex


def union(shapes):
    if not shapes:
        return None

    shapes = list(shapes)
    if len(shapes) == 0:
        return None
    if len(shapes) == 1:
        return shapes[0]

    crs = shapes[0].crs
    shapes = [s.transform(crs) for s in shapes]
    geo = shapely.ops.unary_union([s.geo for s in shapes])
    return Shape(geo, crs)


_DEFAULT_POINT_BUFFER_RESOLUTION = 6


def buffer_point(sh, tolerance, resolution=_DEFAULT_POINT_BUFFER_RESOLUTION):
    if sh.type != 'Point':
        return
    if not tolerance:
        return
    return Shape(sh.geo.buffer(tolerance, resolution), sh.crs)


class Shape(t.ShapeInterface):
    crs = ''
    geo = None

    def __init__(self, geo, crs):
        self.crs = gws.gis.proj.as_epsg(crs)
        self.crs_code = self.crs.split(':')[1]
        self.geo = geo

    @property
    def type(self):
        return self.geo.type

    @property
    def props(self):
        return {
            'crs': self.crs,
            'geometry': shapely.geometry.mapping(self.geo),
        }

    @property
    def wkb(self):
        return self.geo.wkb

    @property
    def wkb_hex(self):
        return self.geo.wkb_hex

    @property
    def wkt(self):
        return self.geo.wkt

    @property
    def bounds(self):
        return self.geo.bounds

    def tolerance_buffer(self, tolerance, resolution=_DEFAULT_POINT_BUFFER_RESOLUTION):
        if not tolerance:
            return self
        return Shape(self.geo.buffer(tolerance, resolution), self.crs)

    def transform(self, to_crs):
        if gws.gis.proj.equal(to_crs, self.crs):
            return self
        return Shape(
            gws.gis.proj.transform(self.geo, self.crs, to_crs),
            to_crs)
