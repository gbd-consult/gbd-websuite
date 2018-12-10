import re

import gdal

import gws
import gws.gis.feature
import gws.gis.shape
import gws.ows.types


def parse(s, first_el, **kwargs):
    if 'gml' not in first_el.namespaces and 'gmlx' not in first_el.namespaces:
        return None

    tmp = '/vsimem/' + gws.random_string(64) + '.xml'
    gdal.FileFromMemBuffer(tmp, s)

    ds = gdal.OpenEx(tmp, open_options=[
        'SWAP_COORDINATES=%s' % ('YES' if kwargs.get('invert_axis') else 'NO'),
        'DOWNLOAD_SCHEMA=NO'
    ])

    if not ds:
        gws.log.error('gdal.gml driver failed')
        gdal.Unlink(tmp)
        return None

    fs = list(_features_from_gdal(ds))
    ds = None
    gdal.Unlink(tmp)

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
            yield gws.gis.feature.Feature({
                'uid': feature.GetFID(),
                'category': layer_name,
                'shape': _shape_from_gdal(feature.GetGeomFieldRef(0)),
                'attributes': atts
            })


def _shape_from_gdal(geom):
    if not geom:
        return
    sr = geom.GetSpatialReference()
    if not sr:
        return
    crs = sr.GetAuthorityName(None) + ':' + sr.GetAuthorityCode(None)
    wkt = geom.ExportToWkt()
    return gws.gis.shape.from_wkt(wkt, crs)
