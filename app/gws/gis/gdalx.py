import os
import contextlib
import datetime

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import gws
import gws.gis.shape
import gws.types as t


class Error(gws.Error):
    pass


def drivers():
    ls = []
    for n in range(gdal.GetDriverCount()):
        driver = gdal.GetDriver(n)
        ls.append(driver.GetDescription())
    return ls


def open(path, mode, **opts) -> 'DataSet':
    if mode == 'w':
        # @TODO guess from extension
        driver_name = opts.pop('driver')
        driver = ogr.GetDriverByName(driver_name)
        if not driver:
            raise Error(f'driver not found {driver_name!r}')
        gd = driver.CreateDataSource(path, **opts)
        if gd is None:
            raise Error(f'cannot create {path!r}')
        return DataSet(gd)

    if not os.path.isfile(path):
        raise Error(f'file not found {path!r}')

    flags = gdal.OF_VERBOSE_ERROR + (gdal.OF_UPDATE if mode == 'a' else gdal.OF_READONLY)
    gd = gdal.OpenEx(path, flags, **opts)
    if gd is None:
        raise Error(f'cannot open {path!r}')
    return DataSet(gd)


class SourceFeature(t.Data):
    attributes: dict
    shape: t.Optional[t.IShape]
    layerName: t.Optional[str]
    uid: str


##

_type_to_ogr = {
    'bool': ogr.OFTInteger,
    'bytes': ogr.OFTBinary,
    'date': ogr.OFTDate,
    'datetime': ogr.OFTDateTime,
    'float': ogr.OFTReal,
    'floatlist': ogr.OFTRealList,
    'int': ogr.OFTInteger,
    'intlist': ogr.OFTIntegerList,
    'bigint': ogr.OFTInteger64,
    'bigintlist': ogr.OFTInteger64List,
    'str': ogr.OFTString,
    'strlist': ogr.OFTStringList,
    'time': ogr.OFTTime,
    'timestamp': ogr.OFTTime,
}

_ogr_to_type = {v: idx for idx, v in _type_to_ogr.items()}

_geom_to_ogr = {
    'CURVE': ogr.wkbCurve,
    'GEOMCOLLECTION': ogr.wkbGeometryCollection,
    'LINESTRING': ogr.wkbLineString,
    'MULTICURVE': ogr.wkbMultiCurve,
    'MULTILINESTRING': ogr.wkbMultiLineString,
    'MULTIPOINT': ogr.wkbMultiPoint,
    'MULTIPOLYGON': ogr.wkbMultiPolygon,
    'MULTISURFACE': ogr.wkbMultiSurface,
    'POINT': ogr.wkbPoint,
    'POLYGON': ogr.wkbPolygon,
    'POLYHEDRALSURFACE': ogr.wkbPolyhedralSurface,
    'SURFACE': ogr.wkbSurface,
}

_ogr_to_geom = {v: idx for idx, v in _geom_to_ogr.items()}

gdal.UseExceptions()


class DataSet:
    def __init__(self, gd: gdal.Dataset):
        self.gdDataset: gdal.Dataset = gd
        driver: gdal.Driver = self.gdDataset.GetDriver()
        self.driverName = driver.GetDescription()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gdDataset = None
        return False

    def create_layer(self, name, columns: dict, geom_type: str = None, srid: int = 0, overwrite=False):
        gtype = _geom_to_ogr[geom_type] if geom_type else ogr.wkbUnknown
        srid = srid or 3857

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(srid)

        opts = []
        if overwrite:
            opts.append('OVERWRITE=YES')

        la = self.gdDataset.CreateLayer(name, geom_type=gtype, srs=srs, options=opts)
        for col_name, col_type in columns.items():
            la.CreateField(ogr.FieldDefn(col_name, _type_to_ogr[col_type]))

        return Layer(la)

    def layer(self, name):
        la = self.gdDataset.GetLayer(name)
        return Layer(la) if la else None

    @contextlib.contextmanager
    def transaction(self):
        self.gdDataset.StartTransaction()
        try:
            yield self
            self.gdDataset.CommitTransaction()
        except:
            self.gdDataset.RollbackTransaction()
            raise

    def close(self):
        pass


class Layer:
    def __init__(self, layer):
        self.gdLayer: ogr.Layer = layer
        self.gdLayerDefn: ogr.FeatureDefn = self.gdLayer.GetLayerDefn()

    def inspect(self):
        tab = t.Data(
            name=self.gdLayerDefn.GetName(),
            columns=[],
        )

        fid_col = self.gdLayer.GetFIDColumn()
        if fid_col:
            tab.columns.append(t.Data(
                name=fid_col,
                type='int',
                isPrimaryKey=True,
            ))

        for i in range(self.gdLayerDefn.GetFieldCount()):
            fdef: ogr.FieldDefn = self.gdLayerDefn.GetFieldDefn(i)
            gt = fdef.GetType()
            tab.columns.append(t.Data(
                name=fdef.GetName(),
                type=_ogr_to_type[gt],
                gdalType=gt,
            ))

        for i in range(self.gdLayerDefn.GetGeomFieldCount()):
            fdef: ogr.GeomFieldDefn = self.gdLayerDefn.GetGeomFieldDefn(i)
            crs: osr.SpatialReference = fdef.GetSpatialRef()
            gt = fdef.GetType()
            tab.columns.append(t.Data(
                name=fdef.GetName(),
                type='geometry',
                geometryType=_ogr_to_geom[gt],
                geometrySrid=int(crs.GetAuthorityCode(None)),
                isGeometry=True,
            ))

        tab.primaryKeys = [c for c in tab.columns if c.isPrimaryKey]
        tab.geometryColumns = [c for c in tab.columns if c.isGeometry]

        return tab

    def feature_count(self):
        return self.gdLayer.GetFeatureCount()

    def insert_feature(self, sf: SourceFeature) -> int:
        feature = ogr.Feature(self.gdLayerDefn)

        if sf.shape:
            # @TODO parse EWKT
            feature.SetGeometry(ogr.CreateGeometryFromWkt(sf.shape.wkt))

        if sf.uid and isinstance(sf.uid, int):
            feature.SetFID(sf.uid)

        for name, val in sf.attributes.items():
            try:
                feature.SetField(name, val)
            except:
                raise Error(f'field {name!r} cannot be set')

        self.gdLayer.CreateFeature(feature)
        fid = feature.GetFID()
        del feature
        return fid

    def features(self, filter: str = None, default_srid: int = 0, encoding: str = None) -> t.List[SourceFeature]:
        self.gdLayer.SetAttributeFilter(filter or '')

        res = []

        while True:
            feature: ogr.Feature = self.gdLayer.GetNextFeature()
            if not feature:
                break

            sf = SourceFeature(
                uid=feature.GetFID(),
                attributes={},
            )

            for i in range(feature.GetFieldCount()):
                fdef = feature.GetFieldDefnRef(i)
                name = fdef.GetName()
                val = _attr_value(feature, i, fdef, encoding)
                sf.attributes[name] = val

            cnt = feature.GetGeomFieldCount()
            if cnt > 0:
                # NB take the last geom
                # @TODO multigeometry support
                geom = feature.GetGeomFieldRef(cnt - 1)
                if not geom:
                    continue
                srs = geom.GetSpatialReference()
                srid = srs.GetAuthorityCode(None) if srs else default_srid
                wkt = geom.ExportToWkt()
                sf.shape = gws.gis.shape.from_wkt(wkt)

            res.append(sf)

        return res


def _attr_value(feature, idx, fdef, encoding):
    ft = fdef.type

    if ft == ogr.OFTString:
        b = feature.GetFieldAsBinary(idx)
        if encoding:
            return b.decode(encoding)
        return b

    if ft in {ogr.OFTDate, ogr.OFTTime, ogr.OFTDateTime}:
        # python GetFieldAsDateTime appears to use float seconds, as in
        # GetFieldAsDateTime (int i, int *pnYear, int *pnMonth, int *pnDay, int *pnHour, int *pnMinute, float *pfSecond, int *pnTZFlag)
        #
        v = feature.GetFieldAsDateTime(idx)
        try:
            return datetime.datetime(v[0], v[1], v[2], v[3], v[4], int(v[5]))
        except ValueError:
            return

    if ft == ogr.OFSTBoolean:
        return feature.GetFieldAsInteger(idx) != 0
    if ft in {ogr.OFTInteger, ogr.OFTInteger64}:
        return feature.GetFieldAsInteger(idx)
    if ft in {ogr.OFTIntegerList, ogr.OFTInteger64List}:
        return feature.GetFieldAsIntegerList(idx)
    if ft in {ogr.OFTReal, ogr.OFSTFloat32}:
        return feature.GetFieldAsDouble(idx)
    if ft == ogr.OFTRealList:
        return feature.GetFieldAsDoubleList(idx)
    if ft == ogr.OFTBinary:
        return feature.GetFieldAsBinary(idx)
