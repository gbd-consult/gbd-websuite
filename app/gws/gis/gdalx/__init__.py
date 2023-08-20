"""GDAL wrapper."""

import os
import contextlib
import datetime

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import gws
import gws.base.shape
import gws.gis.crs
import gws.types as t


class Error(gws.Error):
    pass


def drivers():
    """Enumerate GDAL drivers."""

    ls = []
    for n in range(gdal.GetDriverCount()):
        driver = gdal.GetDriver(n)
        ls.append(driver.GetDescription())

    return sorted(ls)


def open(path, mode, driver=None, **opts) -> 'DataSet':
    """Open a path and return a DataSet object.

    Args:
        path: file path
        mode: 'r' (read), 'a' (update), 'w' (create for writing)
        driver: driver name, if omitted, will be suggested from the path extension
        opts: options for gdal.OpenEx/CreateDataSource

    Returns:
        DataSet object.

    """

    gdal.UseExceptions()

    if mode == 'w':
        if not driver:
            driver = _driver_name_from_extension(path.split('.')[-1])
        if not driver:
            raise Error(f'no driver found for {path!r}')
        drv = ogr.GetDriverByName(driver)
        if not drv:
            raise Error(f'driver not found {driver!r}')
        gd = drv.CreateDataSource(path, **opts)
        if gd is None:
            raise Error(f'cannot create {path!r}')
        return DataSet(path, gd)

    if not os.path.isfile(path):
        raise Error(f'file not found {path!r}')

    flags = gdal.OF_VERBOSE_ERROR + (gdal.OF_UPDATE if mode == 'a' else gdal.OF_READONLY)
    gd = gdal.OpenEx(path, flags, **opts)
    if gd is None:
        raise Error(f'cannot open {path!r}')
    return DataSet(path, gd)


class DataSet:
    gdDataset: gdal.Dataset
    gdDriver: gdal.Driver
    path: str
    driverName: str

    def __init__(self, path: str, gd: gdal.Dataset):
        self.path = path
        self.gdDataset = gd
        self.gdDriver = self.gdDataset.GetDriver()
        self.driverName = self.gdDriver.GetDescription()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

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
        setattr(self, 'gdDataset', None)

    def create_layer(
            self,
            name: str,
            columns: dict[str, gws.AttributeType],
            geometry_type: gws.GeometryType = None,
            crs: gws.ICrs = None,
            overwrite=False,
            *options,
    ) -> 'Layer':
        opts = list(options)
        if overwrite:
            opts.append('OVERWRITE=YES')

        gd_layer = self.gdDataset.CreateLayer(
            name,
            geom_type=_GEOM_TO_OGR[geometry_type] if geometry_type else ogr.wkbUnknown,
            srs=_srs(crs.srid if crs else 3857) if geometry_type else None,
            options=opts,
        )
        for col_name, col_type in columns.items():
            gd_layer.CreateField(ogr.FieldDefn(col_name, _ATTR_TO_OGR[col_type]))

        return Layer(gd_layer)

    def layers(self):
        cnt = self.gdDataset.GetLayerCount()
        return [Layer(self.gdDataset.GetLayerByIndex(n)) for n in range(cnt)]

    def layer(self, name) -> t.Optional['Layer']:
        gd_layer = self.gdDataset.GetLayer(name)
        return Layer(gd_layer) if gd_layer else None

    def describe_layer(self, name) -> t.Optional[gws.DataSetDescription]:
        la = self.layer(name)
        if la:
            return la.describe()


class Layer:
    name: str
    gdLayer: ogr.Layer
    gdLayerDefn: ogr.FeatureDefn

    def __init__(self, gd_layer):
        self.gdLayer = gd_layer
        self.gdLayerDefn = self.gdLayer.GetLayerDefn()
        self.name = self.gdLayerDefn.GetName()

    def describe(self) -> gws.DataSetDescription:
        desc = gws.DataSetDescription(
            name=self.name,
            fullName=self.name,
            schema='',
            columns={},
            keyNames=[],
            geometryName='',
            geometryType='',
            geometrySrid='',
        )

        cols = []

        fid_col = self.gdLayer.GetFIDColumn()
        if fid_col:
            cols.append(gws.ColumnDescription(
                name=fid_col,
                type=_OGR_TO_ATTR[ogr.OFTInteger],
                nativeType=ogr.OFTInteger,
                isPrimaryKey=True,
                columnIndex=0,
            ))

        for i in range(self.gdLayerDefn.GetFieldCount()):
            fdef: ogr.FieldDefn = self.gdLayerDefn.GetFieldDefn(i)
            typ = fdef.GetType()
            if typ not in _OGR_TO_ATTR:
                continue
            cols.append(gws.ColumnDescription(
                name=fdef.GetName(),
                type=_OGR_TO_ATTR[typ],
                nativeType=typ,
                columnIndex=i,
            ))

        for i in range(self.gdLayerDefn.GetGeomFieldCount()):
            fdef: ogr.GeomFieldDefn = self.gdLayerDefn.GetGeomFieldDefn(i)
            crs: osr.SpatialReference = fdef.GetSpatialRef()
            typ = fdef.GetType()
            if typ not in _OGR_TO_GEOM:
                continue
            cols.append(gws.ColumnDescription(
                name=fdef.GetName(),
                type=gws.AttributeType.geometry,
                nativeType=typ,
                columnIndex=i,
                geometryType=_OGR_TO_GEOM[typ],
                geometrySrid=int(crs.GetAuthorityCode(None)),
            ))

        desc.columns = {c.name: c for c in cols}

        for c in cols:
            if c.isPrimaryKey:
                desc.keyNames.append(c.name)

        for c in cols:
            # NB take the last geom
            if c.geometryType:
                desc.geometryName = c.name
                desc.geometryType = c.geometryType
                desc.geometrySrid = c.geometrySrid

        return desc

    def insert(self, fds: list[gws.FeatureData], encoding: str = None) -> list[int]:
        desc = self.describe()
        fids = []

        for fd in fds:
            gd_feature = ogr.Feature(self.gdLayerDefn)
            if desc.geometryType:
                if fd.shape:
                    gd_feature.SetGeometry(
                        ogr.CreateGeometryFromWkt(
                            fd.shape.to_wkt(),
                            _srs(fd.shape.crs.srid)))
                elif fd.wkt:
                    # NB using default CRS
                    gd_feature.SetGeometry(ogr.CreateGeometryFromWkt(fd.wkt))

            if fd.uid and isinstance(fd.uid, int):
                gd_feature.SetFID(fd.uid)

            for col in desc.columns.values():
                if col.geometryType or col.isPrimaryKey:
                    continue
                val = fd.attributes.get(col.name)
                if val is None:
                    continue
                try:
                    _attr_to_ogr(gd_feature, int(col.nativeType), col.columnIndex, val, encoding)
                except Exception as exc:
                    raise Error(f'field {col.name!r} cannot be set (value={val!r})') from exc

            self.gdLayer.CreateFeature(gd_feature)
            fids.append(gd_feature.GetFID())

        return fids

    def count(self, force=False):
        return self.gdLayer.GetFeatureCount(force=1 if force else 0)

    def get_all(self, default_srid: int = 0, encoding: str = None) -> list[gws.FeatureData]:
        fds = []
        self.gdLayer.ResetReading()
        while True:
            gd_feature = self.gdLayer.GetNextFeature()
            if not gd_feature:
                break
            fds.append(self._feature_data(gd_feature, default_srid, encoding))
        return fds

    def get_one(self, fid: int, default_srid: int = 0, encoding: str = None) -> t.Optional[gws.FeatureData]:
        gd_feature = self.gdLayer.GetFeature(fid)
        if gd_feature:
            return self._feature_data(gd_feature, default_srid, encoding)

    def _feature_data(self, gd_feature, default_srid, encoding):
        fd = gws.FeatureData(
            attributes={},
            shape=None,
            layerName=self.name,
            uid=str(gd_feature.GetFID()),
        )

        for i in range(gd_feature.GetFieldCount()):
            gd_field_defn: ogr.FieldDefn = gd_feature.GetFieldDefnRef(i)
            name = gd_field_defn.GetName()
            val = _attr_from_ogr(gd_feature, gd_field_defn.type, i, encoding)
            fd.attributes[name] = val

        cnt = gd_feature.GetGeomFieldCount()
        if cnt > 0:
            # NB take the last geom
            # @TODO multigeometry support
            gd_geom_defn = gd_feature.GetGeomFieldRef(cnt - 1)
            if gd_geom_defn:
                srs = gd_geom_defn.GetSpatialReference()
                srid = srs.GetAuthorityCode(None) if srs else default_srid
                wkt = gd_geom_defn.ExportToWkt()
                fd.shape = gws.base.shape.from_wkt(wkt, gws.gis.crs.get(srid))

        return fd


##

_ext_to_driver_name = {}


def _driver_name_from_extension(s):
    if not _ext_to_driver_name:
        for n in range(gdal.GetDriverCount()):
            drv = gdal.GetDriver(n)
            name = drv.GetDescription()
            exts = drv.GetMetadataItem(gdal.DMD_EXTENSIONS)
            if exts:
                # NB exts is a space delimited string
                for e in exts.split():
                    _ext_to_driver_name[e] = name

    return _ext_to_driver_name.get(s)


_srs_cache = {}


def _srs(srid):
    if srid not in _srs_cache:
        _srs_cache[srid] = osr.SpatialReference()
        _srs_cache[srid].ImportFromEPSG(srid)
    return _srs_cache[srid]


def _attr_from_ogr(gd_feature: ogr.Feature, gtype: int, idx: int, encoding: str = 'utf8'):
    if gd_feature.IsFieldNull(idx):
        return None

    if gtype == ogr.OFTString:
        b = gd_feature.GetFieldAsBinary(idx)
        if encoding:
            return b.decode(encoding)
        return b

    if gtype in {ogr.OFTDate, ogr.OFTTime, ogr.OFTDateTime}:
        # python GetFieldAsDateTime appears to use float seconds, as in
        # GetFieldAsDateTime (int i, int *pnYear, int *pnMonth, int *pnDay, int *pnHour, int *pnMinute, float *pfSecond, int *pnTZFlag)
        #
        v = gd_feature.GetFieldAsDateTime(idx)
        try:
            return datetime.datetime(v[0], v[1], v[2], v[3], v[4], int(v[5]))
        except ValueError:
            return

    if gtype == ogr.OFSTBoolean:
        return gd_feature.GetFieldAsInteger(idx) != 0
    if gtype in {ogr.OFTInteger, ogr.OFTInteger64}:
        return gd_feature.GetFieldAsInteger(idx)
    if gtype in {ogr.OFTIntegerList, ogr.OFTInteger64List}:
        return gd_feature.GetFieldAsIntegerList(idx)
    if gtype in {ogr.OFTReal, ogr.OFSTFloat32}:
        return gd_feature.GetFieldAsDouble(idx)
    if gtype == ogr.OFTRealList:
        return gd_feature.GetFieldAsDoubleList(idx)
    if gtype == ogr.OFTBinary:
        return gd_feature.GetFieldAsBinary(idx)


def _attr_to_ogr(gd_feature: ogr.Feature, gtype: int, idx: int, value, encoding: str = None):
    if gtype in {ogr.OFTDate, ogr.OFTTime, ogr.OFTDateTime}:
        v = t.cast(datetime.datetime, value).isoformat()
        gd_feature.SetField(idx, v)
        return

    if gtype == ogr.OFSTBoolean:
        return gd_feature.SetField(idx, bool(value))
    if gtype in {ogr.OFTInteger, ogr.OFTInteger64}:
        return gd_feature.SetField(idx, int(value))
    if gtype in {ogr.OFTIntegerList, ogr.OFTInteger64List}:
        return gd_feature.SetField(idx, [int(x) for x in value])
    if gtype in {ogr.OFTReal, ogr.OFSTFloat32}:
        return gd_feature.SetField(idx, float(value))
    if gtype == ogr.OFTRealList:
        return gd_feature.SetField(idx, [float(x) for x in value])

    return gd_feature.SetField(idx, value)


_ATTR_TO_OGR = {
    gws.AttributeType.bool: ogr.OFTInteger,
    gws.AttributeType.bytes: ogr.OFTBinary,
    gws.AttributeType.date: ogr.OFTDate,
    gws.AttributeType.datetime: ogr.OFTDateTime,
    gws.AttributeType.float: ogr.OFTReal,
    gws.AttributeType.floatlist: ogr.OFTRealList,
    gws.AttributeType.int: ogr.OFTInteger,
    gws.AttributeType.intlist: ogr.OFTIntegerList,
    gws.AttributeType.str: ogr.OFTString,
    gws.AttributeType.strlist: ogr.OFTStringList,
    gws.AttributeType.time: ogr.OFTTime,
}

_OGR_TO_ATTR = {
    ogr.OFTBinary: gws.AttributeType.bytes,
    ogr.OFTDate: gws.AttributeType.date,
    ogr.OFTDateTime: gws.AttributeType.datetime,
    ogr.OFTReal: gws.AttributeType.float,
    ogr.OFTRealList: gws.AttributeType.floatlist,
    ogr.OFTInteger: gws.AttributeType.int,
    ogr.OFTIntegerList: gws.AttributeType.intlist,
    ogr.OFTInteger64: gws.AttributeType.int,
    ogr.OFTInteger64List: gws.AttributeType.intlist,
    ogr.OFTString: gws.AttributeType.str,
    ogr.OFTStringList: gws.AttributeType.strlist,
    ogr.OFTTime: gws.AttributeType.time,
}

_GEOM_TO_OGR = {
    gws.GeometryType.curve: ogr.wkbCurve,
    gws.GeometryType.geometrycollection: ogr.wkbGeometryCollection,
    gws.GeometryType.linestring: ogr.wkbLineString,
    gws.GeometryType.multicurve: ogr.wkbMultiCurve,
    gws.GeometryType.multilinestring: ogr.wkbMultiLineString,
    gws.GeometryType.multipoint: ogr.wkbMultiPoint,
    gws.GeometryType.multipolygon: ogr.wkbMultiPolygon,
    gws.GeometryType.multisurface: ogr.wkbMultiSurface,
    gws.GeometryType.point: ogr.wkbPoint,
    gws.GeometryType.polygon: ogr.wkbPolygon,
    gws.GeometryType.polyhedralsurface: ogr.wkbPolyhedralSurface,
    gws.GeometryType.surface: ogr.wkbSurface,
}

_OGR_TO_GEOM = {
    ogr.wkbCurve: gws.GeometryType.curve,
    ogr.wkbGeometryCollection: gws.GeometryType.geometrycollection,
    ogr.wkbLineString: gws.GeometryType.linestring,
    ogr.wkbMultiCurve: gws.GeometryType.multicurve,
    ogr.wkbMultiLineString: gws.GeometryType.multilinestring,
    ogr.wkbMultiPoint: gws.GeometryType.multipoint,
    ogr.wkbMultiPolygon: gws.GeometryType.multipolygon,
    ogr.wkbMultiSurface: gws.GeometryType.multisurface,
    ogr.wkbPoint: gws.GeometryType.point,
    ogr.wkbPolygon: gws.GeometryType.polygon,
    ogr.wkbPolyhedralSurface: gws.GeometryType.polyhedralsurface,
    ogr.wkbSurface: gws.GeometryType.surface,
}
