"""GML geometry writer."""

from typing import Optional

import shapely.geometry

import gws
import gws.lib.uom
import gws.lib.xmlx as xmlx
from gws.lib.xmlx import tag

# @TODO PostGis options 2 and 4 (https://postgis.net/docs/ST_AsGML.html)


class _Options(gws.Data):
    version: int
    precision: float
    swapxy: bool
    xmlns: str
    crsName: dict


DEFAULT_VERSION = 3


def shape_to_element(
    shape: gws.Shape,
    version: int = DEFAULT_VERSION,
    coordinate_precision: Optional[int] = None,
    always_xy: bool = False,
    with_xmlns: bool = True,
    with_inline_xmlns: bool = False,
    namespace: Optional[gws.XmlNamespace] = None,
    crs_format: Optional[gws.CrsFormat] = None,
) -> gws.XmlElement:
    """Convert a Shape to a GML geometry element.

    Args:
        shape: A Shape object.
        version: GML version (2 or 3).
        coordinate_precision: The amount of decimal places.
        always_xy: If ``True``, coordinates are assumed to be always in the XY (lon/lat) order.
        with_xmlns: If ``True`` put the elements in the GML namespace.
        with_inline_xmlns: If ``True`` declare the namespace on the geometry element.
        namespace: Use this namespace (default "gml").
        crs_format: Crs format to use (default "url" for version 2 and "urn" for version 3).

    Returns:
        A GML element.
    """

    opts = _Options()
    opts.version = int(version or DEFAULT_VERSION)
    if opts.version not in {2, 3}:
        raise gws.Error(f'unsupported GML version {version!r}')

    crs_format = crs_format or (gws.CrsFormat.url if opts.version == 2 else gws.CrsFormat.urn)
    opts.crsName = {'srsName': shape.crs.to_string(crs_format)}

    opts.swapxy = (shape.crs.axis_for_format(crs_format) == gws.Axis.yx) and not always_xy
    opts.precision = coordinate_precision if coordinate_precision is not None else gws.lib.uom.DEFAULT_PRECISION[shape.crs.uom]

    opts.clarkPrefix = ''
    ns = None
    if with_xmlns:
        ns = namespace or (xmlx.namespace.c.GML_2 if opts.version == 2 else xmlx.namespace.c.GML)
        opts.clarkPrefix = '{' + ns.uri + '}'

    geom: shapely.geometry.base.BaseGeometry = getattr(shape, 'geom')
    fn = _tag2 if opts.version == 2 else _tag3

    # OGC 07-036r1 10.1.4.1
    # If no srsName attribute is given, the CRS shall be specified as part of the larger context this geometry element is part of...
    # NOTE It is expected that the attribute will be specified at the direct position level only in rare cases.

    el = fn(geom, opts)
    if ns and with_inline_xmlns:
        el.declare(ns)

    return el


def _point2(geom, opts):
    return tag(f'{opts.clarkPrefix}Point', opts.crsName, _coordinates(geom, opts))


def _point3(geom, opts):
    return tag(f'{opts.clarkPrefix}Point', opts.crsName, _pos(geom, opts))


def _linestring2(geom, opts):
    return tag(f'{opts.clarkPrefix}LineString', opts.crsName, _coordinates(geom, opts))


def _linestring3(geom, opts):
    return tag(
        f'{opts.clarkPrefix}Curve',
        opts.crsName,
        tag(
            f'{opts.clarkPrefix}segments',
            tag(
                f'{opts.clarkPrefix}LineStringSegment',
                _pos_list(geom, opts),
            ),
        ),
    )


def _polygon2(geom, opts):
    return tag(
        f'{opts.clarkPrefix}Polygon',
        opts.crsName,
        tag(
            f'{opts.clarkPrefix}outerBoundaryIs',
            tag(
                f'{opts.clarkPrefix}LinearRing',
                _coordinates(geom.exterior, opts),
            ),
        ),
        [
            tag(
                f'{opts.clarkPrefix}innerBoundaryIs',
                tag(
                    f'{opts.clarkPrefix}LinearRing',
                    _coordinates(interior, opts),
                ),
            )
            for interior in geom.interiors
        ],
    )


def _polygon3(geom, opts):
    return tag(
        f'{opts.clarkPrefix}Polygon',
        opts.crsName,
        tag(
            f'{opts.clarkPrefix}exterior',
            tag(
                f'{opts.clarkPrefix}LinearRing',
                _pos_list(geom.exterior, opts),
            ),
        ),
        [
            tag(
                f'{opts.clarkPrefix}interior',
                tag(
                    f'{opts.clarkPrefix}LinearRing',
                    _pos_list(interior, opts),
                ),
            )
            for interior in geom.interiors
        ],
    )


def _multipoint2(geom, opts):
    return tag(f'{opts.clarkPrefix}MultiPoint', opts.crsName, [tag(f'{opts.clarkPrefix}pointMember', _tag2(p, opts)) for p in geom.geoms])


def _multipoint3(geom, opts):
    return tag(f'{opts.clarkPrefix}MultiPoint', opts.crsName, [tag(f'{opts.clarkPrefix}pointMember', _tag3(p, opts)) for p in geom.geoms])


def _multilinestring2(geom, opts):
    return tag(f'{opts.clarkPrefix}MultiLineString', opts.crsName, [tag(f'{opts.clarkPrefix}lineStringMember', _tag2(p, opts)) for p in geom.geoms])


def _multilinestring3(geom, opts):
    return tag(f'{opts.clarkPrefix}MultiCurve', opts.crsName, [tag(f'{opts.clarkPrefix}curveMember', _tag3(p, opts)) for p in geom.geoms])


def _multipolygon2(geom, opts):
    return tag(f'{opts.clarkPrefix}MultiPolygon', opts.crsName, [tag(f'{opts.clarkPrefix}polygonMember', _tag2(p, opts)) for p in geom.geoms])


def _multipolygon3(geom, opts):
    return tag(f'{opts.clarkPrefix}MultiSurface', opts.crsName, [tag(f'{opts.clarkPrefix}surfaceMember', _tag3(p, opts)) for p in geom.geoms])


def _geometrycollection2(geom, opts):
    return tag(f'{opts.clarkPrefix}MultiGeometry', opts.crsName, [tag(f'{opts.clarkPrefix}geometryMember', _tag2(p, opts)) for p in geom.geoms])


def _geometrycollection3(geom, opts):
    return tag(f'{opts.clarkPrefix}MultiGeometry', opts.crsName, [tag(f'{opts.clarkPrefix}geometryMember', _tag3(p, opts)) for p in geom.geoms])


def _pos(geom, opts):
    return tag(f'{opts.clarkPrefix}pos', {'srsDimension': 2}, _pos_list_content(geom, opts))


def _pos_list(geom, opts):
    return tag(f'{opts.clarkPrefix}posList', {'srsDimension': 2}, _pos_list_content(geom, opts))


def _pos_list_content(geom, opts):
    cs = []

    for x, y in geom.coords:
        x = int(x) if opts.precision == 0 else round(x, opts.precision)
        y = int(y) if opts.precision == 0 else round(y, opts.precision)
        if opts.swapxy:
            x, y = y, x
        cs.append(str(x))
        cs.append(str(y))

    return ' '.join(cs)


def _coordinates(geom, opts):
    cs = []

    for x, y in geom.coords:
        x = int(x) if opts.precision == 0 else round(x, opts.precision)
        y = int(y) if opts.precision == 0 else round(y, opts.precision)
        if opts.swapxy:
            x, y = y, x
        cs.append(str(x) + ',' + str(y))

    return tag(f'{opts.clarkPrefix}coordinates', {'decimal': '.', 'cs': ',', 'ts': ' '}, ' '.join(cs))


_FNS_2 = {
    'Point': _point2,
    'LineString': _linestring2,
    'Polygon': _polygon2,
    'MultiPoint': _multipoint2,
    'MultiLineString': _multilinestring2,
    'MultiPolygon': _multipolygon2,
    'GeometryCollection': _geometrycollection2,
}

_FNS_3 = {
    'Point': _point3,
    'LineString': _linestring3,
    'Polygon': _polygon3,
    'MultiPoint': _multipoint3,
    'MultiLineString': _multilinestring3,
    'MultiPolygon': _multipolygon3,
    'GeometryCollection': _geometrycollection3,
}


def _tag2(geom, opts):
    typ = geom.geom_type
    fn = _FNS_2.get(typ)
    if fn:
        return fn(geom, opts)
    raise gws.Error(f'cannot convert geometry type {typ!r} to GML')


def _tag3(geom, opts):
    typ = geom.geom_type
    fn = _FNS_3.get(typ)
    if fn:
        return fn(geom, opts)
    raise gws.Error(f'cannot convert geometry type {typ!r} to GML')
