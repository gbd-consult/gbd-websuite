"""GDAL wrapper."""

from typing import Optional, cast

import contextlib

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import gws
import gws.base.shape
import gws.gis.crs
import gws.lib.datetimex as datetimex


class Error(gws.Error):
    pass


class DriverInfo(gws.Data):
    index: int
    name: str
    longName: str
    metaData: dict


class DriverVariant(gws.Enum):
    raster = 'raster'
    vector = 'vector'


class _DriverInfoCache(gws.Data):
    infos: list[DriverInfo]
    extToName: dict
    vectorNames: set[str]
    rasterNames: set[str]


def drivers() -> list[DriverInfo]:
    """Enumerate GDAL drivers."""

    di = _fetch_driver_infos()
    return di.infos


def open(
        path: str,
        mode: str,
        driver: str = '',
        variant: Optional[DriverVariant] = None,
        encoding: str = 'utf8',
        default_crs: Optional[gws.Crs] = None,
        **opts
) -> 'DataSet':
    """Open a path and return a DataSet object.

    Args:
        path: File path.
        mode: 'r' (read), 'a' (update), 'w' (create for writing)
        driver: Driver name, if omitted, will be suggested from the path extension.
        variant: 'raster' or 'vector'  for dual drivers, like Geopackage.
        encoding: If not None, strings will be automatically decoded.
        default_crs: Default CRS for geometries (fallback to Webmercator).
        opts: Options for gdal.OpenEx/CreateDataSource.

    Returns:
        DataSet object.

    """

    gdal.UseExceptions()

    drv = _driver_from_args(path, driver, variant)
    default_crs = default_crs or gws.gis.crs.WEBMERCATOR

    if mode == 'w':
        gd = drv.CreateDataSource(path, **opts)
        if gd is None:
            raise Error(f'cannot create {path!r}')
        return DataSet(path, gd, encoding, default_crs)

    if not gws.u.is_file(path):
        raise Error(f'file not found {path!r}')

    flags = gdal.OF_VERBOSE_ERROR + (gdal.OF_UPDATE if mode == 'a' else gdal.OF_READONLY)
    if variant == DriverVariant.raster:
        flags += gdal.OF_RASTER
    if variant == DriverVariant.vector:
        flags += gdal.OF_VECTOR
    gd = gdal.OpenEx(path, flags, **opts)
    if gd is None:
        raise Error(f'cannot open {path!r}')
    return DataSet(path, gd, encoding, default_crs)


def open_image(image: gws.Image, bounds: gws.Bounds) -> 'DataSet':
    gdal.UseExceptions()

    drv = gdal.GetDriverByName('MEM')
    img_array = image.to_array()
    band_count = img_array.shape[2]

    gd = drv.Create('', img_array.shape[1], img_array.shape[0], band_count, gdal.GDT_Byte)
    for band in range(band_count):
        gd.GetRasterBand(band + 1).WriteArray(img_array[:, :, band])

    ext = bounds.extent

    src_res_x = (ext[2] - ext[0]) / gd.RasterXSize
    src_res_y = (ext[1] - ext[3]) / gd.RasterYSize

    src_transform = (
        ext[0],
        src_res_x,
        0,
        ext[3],
        0,
        src_res_y,
    )

    gd.SetGeoTransform(src_transform)
    gd.SetSpatialRef(_srs_from_srid(bounds.crs.srid))

    return DataSet('', gd)


def create_copy(path: str, ds: 'DataSet', driver: str = '', strict=False, **opts) -> 'DataSet':
    gdal.UseExceptions()

    drv = _driver_from_args(path, driver, variant=DriverVariant.raster)
    gd = drv.CreateCopy(path, ds.gdDataset, 1 if strict else 0, **opts)
    gd.SetMetadata(ds.gdDataset.GetMetadata())
    gd.FlushCache()
    return DataSet(path, gd)


class DataSet:
    gdDataset: gdal.Dataset
    gdDriver: gdal.Driver
    path: str
    driverName: str
    defaultCrs: gws.Crs
    encoding: str

    def __init__(self, path, gd, encoding='utf8', default_crs: Optional[gws.Crs] = None):
        self.path = path
        self.gdDataset = gd
        self.gdDriver = self.gdDataset.GetDriver()
        self.driverName = self.gdDriver.GetDescription()
        self.encoding = encoding
        self.defaultCrs = default_crs or gws.gis.crs.WEBMERCATOR

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
        self.gdDataset.FlushCache()
        setattr(self, 'gdDataset', None)

    def create_layer(
            self,
            name: str,
            columns: dict[str, gws.AttributeType],
            geometry_type: gws.GeometryType = None,
            crs: gws.Crs = None,
            overwrite=False,
            *options,
    ) -> 'Layer':
        opts = list(options)
        if overwrite:
            opts.append('OVERWRITE=YES')

        geom_type = ogr.wkbUnknown
        srs = None

        if geometry_type:
            geom_type = _GEOM_TO_OGR.get(geometry_type)
            crs = crs or self.defaultCrs
            srs = _srs_from_srid(crs.srid)

        gd_layer = self.gdDataset.CreateLayer(
            name,
            geom_type=geom_type,
            srs=srs,
            options=opts,
        )
        for col_name, col_type in columns.items():
            gd_layer.CreateField(ogr.FieldDefn(col_name, _ATTR_TO_OGR[col_type]))

        return Layer(self, gd_layer)

    def layers(self):
        cnt = self.gdDataset.GetLayerCount()
        return [Layer(self, self.gdDataset.GetLayerByIndex(n)) for n in range(cnt)]

    def layer(self, name_or_index: str | int) -> Optional['Layer']:
        gd_layer = self.gdDataset.GetLayer(name_or_index)
        return Layer(self, gd_layer) if gd_layer else None


class Layer:
    name: str
    gdLayer: ogr.Layer
    gdDefn: ogr.FeatureDefn
    encoding: str
    defaultCrs: gws.Crs

    def __init__(self, ds: DataSet, gd_layer: ogr.Layer):
        self.gdLayer = gd_layer
        self.gdDefn = self.gdLayer.GetLayerDefn()
        self.name = self.gdDefn.GetName()
        self.encoding = ds.encoding
        self.defaultCrs = ds.defaultCrs

    def describe(self) -> gws.DataSetDescription:
        desc = gws.DataSetDescription(
            columns=[],
            columnMap={},
            fullName=self.name,
            geometryName='',
            geometrySrid=0,
            geometryType='',
            name=self.name,
            schema='',
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

        for i in range(self.gdDefn.GetFieldCount()):
            fdef: ogr.FieldDefn = self.gdDefn.GetFieldDefn(i)
            typ = fdef.GetType()
            if typ not in _OGR_TO_ATTR:
                continue
            cols.append(gws.ColumnDescription(
                name=fdef.GetName(),
                type=_OGR_TO_ATTR[typ],
                nativeType=typ,
                columnIndex=i,
            ))

        for i in range(self.gdDefn.GetGeomFieldCount()):
            fdef: ogr.GeomFieldDefn = self.gdDefn.GetGeomFieldDefn(i)
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

        desc.columns = cols
        desc.columnMap = {c.name: c for c in cols}

        for c in cols:
            # NB take the last geom
            if c.geometryType:
                desc.geometryName = c.name
                desc.geometryType = c.geometryType
                desc.geometrySrid = c.geometrySrid

        return desc

    def insert(self, records: list[gws.FeatureRecord]) -> list[int]:
        desc = self.describe()
        fids = []

        for rec in records:
            gd_feature = ogr.Feature(self.gdDefn)
            if desc.geometryType and rec.shape:
                gd_feature.SetGeometry(
                    ogr.CreateGeometryFromWkt(
                        rec.shape.to_wkt(),
                        _srs_from_srid(rec.shape.crs.srid)
                    ))

            if rec.uid and isinstance(rec.uid, int):
                gd_feature.SetFID(rec.uid)

            for col in desc.columns:
                if col.geometryType or col.isPrimaryKey:
                    continue
                val = rec.attributes.get(col.name)
                if val is None:
                    continue
                try:
                    _attr_to_ogr(gd_feature, int(col.nativeType), col.columnIndex, val, self.encoding)
                except Exception as exc:
                    raise Error(f'field {col.name!r} cannot be set (value={val!r})') from exc

            self.gdLayer.CreateFeature(gd_feature)
            fids.append(gd_feature.GetFID())

        return fids

    def count(self, force=False):
        return self.gdLayer.GetFeatureCount(force=1 if force else 0)

    def get_all(self) -> list[gws.FeatureRecord]:
        records = []

        self.gdLayer.ResetReading()

        while True:
            gd_feature = self.gdLayer.GetNextFeature()
            if not gd_feature:
                break
            records.append(self._feature_record(gd_feature))

        return records

    def get(self, fid: int) -> Optional[gws.FeatureRecord]:
        gd_feature = self.gdLayer.GetFeature(fid)
        if gd_feature:
            return self._feature_record(gd_feature)

    def _feature_record(self, gd_feature):
        rec = gws.FeatureRecord(
            attributes={},
            shape=None,
            meta={'layerName': self.name},
            uid=str(gd_feature.GetFID()),
        )

        for i in range(gd_feature.GetFieldCount()):
            gd_field_defn: ogr.FieldDefn = gd_feature.GetFieldDefnRef(i)
            name = gd_field_defn.GetName()
            val = _attr_from_ogr(gd_feature, gd_field_defn.type, i, self.encoding)
            rec.attributes[name] = val

        cnt = gd_feature.GetGeomFieldCount()
        if cnt > 0:
            # NB take the last geom
            # @TODO multigeometry support
            gd_geom_defn = gd_feature.GetGeomFieldRef(cnt - 1)
            if gd_geom_defn:
                srs = gd_geom_defn.GetSpatialReference()
                if srs:
                    srid = srs.GetAuthorityCode(None)
                    crs = gws.gis.crs.get(srid)
                else:
                    crs = self.defaultCrs
                wkt = gd_geom_defn.ExportToWkt()
                rec.shape = gws.base.shape.from_wkt(wkt, crs)

        return rec


##


def _driver_from_args(path, driver_name, variant):
    di = _fetch_driver_infos()

    if not driver_name:
        ext = path.split('.')[-1]
        names = di.extToName.get(ext)
        if not names:
            raise Error(f'no default driver found for {path!r}')
        if len(names) > 1:
            raise Error(f'multiple drivers found for {path!r}')
        driver_name = names[0]

    is_vector = driver_name in di.vectorNames
    is_raster = driver_name in di.rasterNames

    if variant == DriverVariant.raster:
        if not is_raster:
            raise Error(f'driver {driver_name!r} is not raster')
        return gdal.GetDriverByName(driver_name)

    if variant == DriverVariant.vector:
        if not is_vector:
            raise Error(f'driver {driver_name!r} is not vector')
        return ogr.GetDriverByName(driver_name)

    # NB prefer raster drivers by default

    if is_raster:
        return gdal.GetDriverByName(driver_name)
    if is_vector:
        return ogr.GetDriverByName(driver_name)

    raise Error(f'driver {driver_name!r} not found')


_di_cache: Optional[_DriverInfoCache] = None


def _fetch_driver_infos() -> _DriverInfoCache:
    global _di_cache

    if _di_cache:
        return _di_cache

    _di_cache = _DriverInfoCache(
        infos=[],
        extToName={},
        vectorNames=set(),
        rasterNames=set(),
    )

    for n in range(gdal.GetDriverCount()):
        drv = gdal.GetDriver(n)
        inf = DriverInfo(
            index=n,
            name=str(drv.ShortName),
            longName=str(drv.LongName),
            metaData=dict(drv.GetMetadata() or {})
        )
        _di_cache.infos.append(inf)

        for e in inf.metaData.get(gdal.DMD_EXTENSIONS, '').split():
            _di_cache.extToName.setdefault(e, []).append(inf.name)
        if inf.metaData.get('DCAP_VECTOR') == 'YES':
            _di_cache.vectorNames.add(inf.name)
        if inf.metaData.get('DCAP_RASTER') == 'YES':
            _di_cache.rasterNames.add(inf.name)

    return _di_cache


_srs_cache = {}


def _srs_from_srid(srid):
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
        sec, msec = str(v[5]).split('.')
        try:
            return datetimex.new(v[0], v[1], v[2], v[3], v[4], int(sec), int(msec), tz=_tzflag_to_tz(v[6]))
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


def _tzflag_to_tz(tzflag):
    # see gdal/ogr/ogrutils.cpp OGRGetISO8601DateTime

    if tzflag == 0 or tzflag == 1:
        return ''
    if tzflag == 100:
        return 'utc'
    if tzflag % 4 != 0:
        # @TODO
        raise Error(f'unsupported timezone {tzflag=}')
    hrs = (100 - tzflag) // 4
    return f'GMT{hrs:+}'


def _attr_to_ogr(gd_feature: ogr.Feature, gtype: int, idx: int, value, encoding):
    if gtype == ogr.OFTDate:
        return gd_feature.SetField(idx, datetimex.to_iso_date_string(value))
    if gtype == ogr.OFTTime:
        return gd_feature.SetField(idx, datetimex.to_iso_time_string(value))
    if gtype == ogr.OFTDateTime:
        return gd_feature.SetField(idx, datetimex.to_iso_string(value))
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
