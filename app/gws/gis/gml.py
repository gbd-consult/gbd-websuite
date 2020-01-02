import osgeo.gdal
import shapely.geometry

import gws
import gws.gis.proj
import gws.gis.feature
import gws.gis.shape
import gws.tools.xml3
import gws.types as t


def tag(*args):
    return args


def envelope_to_extent(el: gws.tools.xml3.Element):
    crs = el.attr('srsName') or 4326
    prj = gws.gis.proj.as_proj(crs)
    if not prj:
        return None, None

    def pair(s):
        s = s.split()
        return float(s[0]), float(s[1])

    try:
        x1, y1 = pair(el.get_text('LowerCorner'))
        x2, y2 = pair(el.get_text('UpperCorner'))
        return prj.epsg, [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]
    except (IndexError, TypeError):
        return None, None


def shape_to_tag(shape: t.IShape, precision=0, invert_axis=False):
    def pos(geom, as_list=True):
        cs = []

        if invert_axis:
            for x, y in geom.coords:
                cs.append(y)
                cs.append(x)
        else:
            for x, y in geom.coords:
                cs.append(x)
                cs.append(y)

        if precision:
            cs = [round(c, precision) for c in cs]
        else:
            cs = [int(c) for c in cs]

        return tag(
            'gml:posList' if as_list else 'gml:pos',
            {'srsDimension': 2},
            ' '.join(str(c) for c in cs))

    def convert(geom, srs=None):
        typ = geom.type

        if typ == 'Point':
            return tag('gml:Point', srs, pos(geom, False))

        if typ == 'LineString':
            return tag('gml:LineString', srs, pos(geom))

        if typ == 'Polygon':
            return tag(
                'gml:Polygon',
                srs,
                tag('gml:exterior', tag('gml:LinearRing', pos(geom.exterior))),
                *[tag('gml:interior', tag('gml:LinearRing', pos(p))) for p in geom.interiors]
            )

        if typ == 'MultiPoint':
            return tag('gml:MultiPoint', srs, *[tag('gml:pointMember', convert(p)) for p in geom])

        if typ == 'MultiLineString':
            return tag('gml:MultiCurve', srs, *[tag('gml:curveMember', convert(p)) for p in geom])

        if typ == 'MultiPolygon':
            return tag('gml:MultiSurface', srs, *[tag('gml:surfaceMember', convert(p)) for p in geom])

    geom: shapely.geometry.base.BaseGeometry = getattr(shape, 'geom')
    return convert(geom, {'srsName': gws.gis.proj.as_urn(shape.crs)})


def features_from_xml(xml, invert_axis=False):
    tmp = '/vsimem/' + gws.random_string(64) + '.xml'
    osgeo.gdal.FileFromMemBuffer(tmp, xml)

    ds = osgeo.gdal.OpenEx(
        tmp,
        allowed_drivers=['gml'],
        open_options=[
            'SWAP_COORDINATES=%s' % ('YES' if invert_axis else 'NO'),
            'DOWNLOAD_SCHEMA=NO'
        ])

    if not ds:
        gws.log.error('gdal.gml driver failed')
        osgeo.gdal.Unlink(tmp)
        return None

    fs = list(_features_from_gdal(ds))
    ds = None
    osgeo.gdal.Unlink(tmp)

    return fs


def _features_from_gdal(ds):
    for n in range(ds.GetLayerCount()):
        layer = ds.GetLayer(n)
        layer_name = layer.GetName()

        for feature in layer:
            atts = {
                feature.GetFieldDefnRef(n).GetName(): feature.GetFieldAsString(n)
                for n in range(feature.GetFieldCount())
            }

            # GetFID returns a number, which is mostly not the case
            uid = feature.GetFID()
            if not uid and 'fid' in atts:
                uid = atts.pop('fid')

            yield gws.gis.feature.Feature(
                uid=uid,
                category=layer_name,
                shape=_shape_from_gdal(feature.GetGeomFieldRef(0)),
                attributes=atts
            )


def _shape_from_gdal(geom):
    if not geom:
        return
    sr = geom.GetSpatialReference()
    if not sr:
        return
    crs = sr.GetAuthorityName(None) + ':' + sr.GetAuthorityCode(None)
    wkt = geom.ExportToWkt()
    return gws.gis.shape.from_wkt(wkt, crs)
