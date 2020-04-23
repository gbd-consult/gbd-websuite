"""GDAL wrappers"""

import contextlib
import osgeo.gdal

import gws
import gws.gis.feature
import gws.gis.shape


@contextlib.contextmanager
def from_string(s, **opts):
    fname = '/vsimem/' + gws.random_string(64)
    osgeo.gdal.FileFromMemBuffer(fname, s)

    ds = osgeo.gdal.OpenEx(fname, **opts)
    yield ds
    del ds
    osgeo.gdal.Unlink(fname)


@contextlib.contextmanager
def from_path(path, **opts):
    ds = osgeo.gdal.OpenEx(path, **opts)
    yield ds
    del ds

def features(ds, crs=None, encoding=None):
    fs = []

    for n in range(ds.GetLayerCount()):
        layer = ds.GetLayer(n)
        layer_name = layer.GetName()

        for feature in layer:
            atts = {}
            for k in range(feature.GetFieldCount()):
                key = feature.GetFieldDefnRef(k).GetName()
                if encoding:
                    val = feature.GetFieldAsBinary(k).decode(encoding)
                else:
                    val = feature.GetFieldAsString(k)
                atts[key] = val

            uid = feature.GetFID()
            if not uid and 'fid' in atts:
                uid = atts.pop('fid')

            fs.append(gws.gis.feature.Feature(
                uid=uid,
                category=layer_name,
                shape=_shape(feature.GetGeomFieldRef(0), crs),
                attributes=atts
            ))

    return fs


def _shape(geom, crs):
    if not geom:
        return
    if not crs:
        sr = geom.GetSpatialReference()
        if not sr:
            return
        crs = _crs(sr)
    if not crs:
        return
    return gws.gis.shape.from_wkt(geom.ExportToWkt(), crs)


def _crs(sr):
    a = sr.GetAuthorityName(None)
    c = sr.GetAuthorityCode(None)
    if a and c:
        return a + ':' + c
    # @TODO deal with non-EPSG codes
