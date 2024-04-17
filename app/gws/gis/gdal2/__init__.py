"""GDAL wrappers"""

import contextlib
import datetime
import osgeo.gdal
import osgeo.ogr

import gws
import gws.base.feature
import gws.base.shape

@contextlib.contextmanager
def from_string(s, **opts):
    fname = '/vsimem/' + gws.u.random_string(64)
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
                fdef = feature.GetFieldDefnRef(k)
                name = fdef.GetName()
                type_val = _value(feature, k, fdef, encoding)
                if type_val:
                    atts[name] = gws.Attribute(name=name, type=type_val[0], value=type_val[1])

            uid = feature.GetFID()
            if not uid and 'fid' in atts:
                uid = atts.pop('fid').value

            fs.append(gws.base.feature.Feature(
                uid=uid,
                category=layer_name,
                shape=_shape(feature.GetGeomFieldRef(0), crs),
                attributes=list(atts.values())
            ))

    return fs


def _value(feature, k, fdef, encoding):
    ft = fdef.type

    if ft == osgeo.ogr.OFTString:
        if encoding:
            return gws.AttributeType.str, feature.GetFieldAsBinary(k).decode(encoding)
        else:
            return gws.AttributeType.str, feature.GetFieldAsString(k)

    if ft in (osgeo.ogr.OFTDate, osgeo.ogr.OFTTime, osgeo.ogr.OFTDateTime):
        # python GetFieldAsDateTime appears to use float seconds, as in
        # GetFieldAsDateTime (int i, int *pnYear, int *pnMonth, int *pnDay, int *pnHour, int *pnMinute, float *pfSecond, int *pnTZFlag)
        #
        v = feature.GetFieldAsDateTime(k)
        try:
            v = datetime.datetime(v[0], v[1], v[2], v[3], v[4], int(v[5]))
            return gws.AttributeType.datetime, v
        except ValueError:
            return

    if ft == osgeo.ogr.OFSTBoolean:
        return gws.AttributeType.bool, feature.GetFieldAsInteger(k) > 0
    if ft in (osgeo.ogr.OFTInteger, osgeo.ogr.OFTInteger64):
        return gws.AttributeType.int, feature.GetFieldAsInteger(k)
    if ft in (osgeo.ogr.OFTIntegerList, osgeo.ogr.OFTInteger64List):
        return gws.AttributeType.intlist, feature.GetFieldAsIntegerList(k)
    if ft in (osgeo.ogr.OFTReal, osgeo.ogr.OFSTFloat32):
        return gws.AttributeType.float, feature.GetFieldAsDouble(k)
    if ft == osgeo.ogr.OFTRealList:
        return gws.AttributeType.floatlist, feature.GetFieldAsDoubleList(k)
    if ft == osgeo.ogr.OFTBinary:
        return gws.AttributeType.bytes, feature.GetFieldAsBinary(k)

    # @TODO

    # osgeo.ogr.OFTStringList
    # osgeo.ogr.OFTWideString
    # osgeo.ogr.OFTWideStringList


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
    return gws.base.shape.from_wkt(geom.ExportToWkt(), crs)


def _crs(sr):
    a = sr.GetAuthorityName(None)
    c = sr.GetAuthorityCode(None)
    if a and c:
        return a + ':' + c
    # @TODO deal with non-EPSG codes
