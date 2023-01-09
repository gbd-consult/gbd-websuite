"""GML geometry writer."""

import shapely.geometry

import gws
import gws.gis.crs
import gws.gis.extent
import gws.base.feature
import gws.base.shape
import gws.lib.xmlx as xmlx
import gws.types as t


# @TODO support GML2
# @TODO PostGis options 2 and 4 (https://postgis.net/docs/ST_AsGML.html)

def shape_to_element(
    shape: gws.IShape,
    precision=0,
    axis: gws.Axis = gws.Axis.xy,
    crs_format: gws.CrsFormat = gws.CrsFormat.urn,
    with_ns='gml'
) -> gws.IXmlElement:
    """Convert a Shape to a GML3 geometry element."""

    geom: shapely.geometry.base.BaseGeometry = getattr(shape, 'geom')
    srs = shape.crs.to_string(crs_format)
    ns = (with_ns + ':') if with_ns else ''

    opts = gws.Data(
        precision=precision,
        axis=axis,
        ns=ns
    )

    return xmlx.tag(*_tag(geom, opts), srsName=srs)


##

def _tag(geom, opts):
    typ = geom.type

    if typ == 'Point':
        return opts.ns + 'Point', _pos(geom, opts, False)

    if typ == 'LineString':
        return opts.ns + 'Curve', (opts.ns + 'segments', (opts.ns + 'LineStringSegment', _pos(geom, opts)))

    if typ == 'Polygon':
        return (
            opts.ns + 'Polygon',
            (opts.ns + 'exterior', (opts.ns + 'LinearRing', _pos(geom.exterior, opts))),
            [
                (opts.ns + 'interior', (opts.ns + 'LinearRing', _pos(r, opts)))
                for r in geom.interiors
            ]
        )

    if typ == 'MultiPoint':
        return opts.ns + 'MultiPoint', [(opts.ns + 'pointMember', _tag(p, opts)) for p in geom]

    if typ == 'MultiLineString':
        return opts.ns + 'MultiCurve', [(opts.ns + 'curveMember', _tag(p, opts)) for p in geom]

    if typ == 'MultiPolygon':
        return opts.ns + 'MultiSurface', [(opts.ns + 'surfaceMember', _tag(p, opts)) for p in geom]

    raise gws.Error(f'cannot convert geometry type {typ!r} to GML')


def _pos(geom, opts, as_list=True):
    cs = []

    if opts.axis == gws.Axis.xy:
        for x, y in geom.coords:
            cs.append(x)
            cs.append(y)
    else:
        for x, y in geom.coords:
            cs.append(y)
            cs.append(x)

    if opts.precision:
        cs = [round(c, opts.precision) for c in cs]
    else:
        cs = [int(c) for c in cs]

    tag = 'posList' if as_list else 'pos'
    return opts.ns + tag, {'srsDimension': 2}, ' '.join(str(c) for c in cs)
