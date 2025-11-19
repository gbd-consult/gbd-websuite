"""GDAL/OGR wrapper."""

from typing import Optional, Iterable, cast

import contextlib
import numpy as np

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import gws
import gws.base.shape
import gws.lib.crs
import gws.lib.bounds
import gws.lib.image
import gws.lib.datetimex as datetimex


class Error(gws.Error):
    pass


class DriverInfo(gws.Data):
    index: int
    name: str
    longName: str
    metaData: dict


class _DriverInfoCache(gws.Data):
    infos: list[DriverInfo]
    extToName: dict
    vectorNames: set[str]
    rasterNames: set[str]


class _DataSetOptions(gws.Data):
    path: str
    mode: str
    driver: str
    encoding: str
    defaultCrs: gws.Crs
    geometryAsText: bool
    gdalOpts: dict


def drivers() -> list[DriverInfo]:
    """Enumerate GDAL drivers."""

    di = _fetch_driver_infos()
    return di.infos


@contextlib.contextmanager
def gdal_config(options: dict):
    """Temporarily set GDAL config options."""

    prev = {}
    for key, value in options.items():
        prev[key] = gdal.GetConfigOption(key)
        gdal.SetConfigOption(key, value)

    try:
        yield
    finally:
        for key, value in prev.items():
            gdal.SetConfigOption(key, value)


def open_raster(
    path: str,
    mode: str = 'r',
    driver: str = '',
    default_crs: Optional[gws.Crs] = None,
    options: dict = None,
) -> 'RasterDataSet':
    """Create a raster DataSet from a path.

    Args:
        path: File path.
        mode: 'r' (=read), 'a' (=update), 'w' (=create/write)
        driver: Driver name, if omitted, will be suggested from the path extension.
        default_crs: Default CRS for geometries (fallback to Webmercator).
        options: Options for gdal.OpenEx/CreateDataSource.
    """

    dso = _DataSetOptions(
        path=path,
        mode=mode,
        driver=driver,
        defaultCrs=default_crs,
        gdalOpts=options or {},
    )

    return cast(RasterDataSet, _open(dso, need_raster=True))


def open_vector(
    path: str,
    mode: str = 'r',
    driver: str = '',
    encoding: Optional[str] = 'utf8',
    default_crs: Optional[gws.Crs] = None,
    geometry_as_text: bool = False,
    options: dict = None,
) -> 'VectorDataSet':
    """Create a vector DataSet from a path.

    Args:
        path: File path.
        mode: 'r' (=read), 'a' (=update), 'w' (=create/write)
        driver: Driver name, if omitted, will be suggested from the path extension.
        encoding: If not None, strings will be automatically decoded.
        default_crs: Default CRS for geometries (fallback to Webmercator).
        geometry_as_text: Don't interpret geometry, extract raw WKT.
        options: Options for gdal.OpenEx/CreateDataSource.


    Returns:
        DataSet object.

    """

    dso = _DataSetOptions(
        path=path,
        mode=mode,
        driver=driver,
        defaultCrs=default_crs,
        encoding=encoding,
        geometryAsText=geometry_as_text,
        gdalOpts=options or {},
    )

    return cast(VectorDataSet, _open(dso, need_raster=False))


def _open(dso: _DataSetOptions, need_raster):
    if not dso.mode:
        dso.mode = 'r'
    if dso.mode not in 'rwa':
        raise Error(f'invalid open mode {dso.mode!r}')

    gdal.UseExceptions()

    drv = _driver_from_args(dso.path, dso.driver, need_raster)
    dso.defaultCrs = dso.defaultCrs or gws.lib.crs.WEBMERCATOR

    if dso.mode == 'w':
        gd = drv.CreateDataSource(dso.path, _option_list(dso.gdalOpts))
        if gd is None:
            raise Error(f'cannot create {dso.path!r}')
        if need_raster:
            return RasterDataSet(dso, gd)
        return VectorDataSet(dso, gd)

    flags = gdal.OF_VERBOSE_ERROR
    if dso.mode == 'r':
        flags += gdal.OF_READONLY
    if dso.mode == 'a':
        flags += gdal.OF_UPDATE
    if need_raster:
        flags += gdal.OF_RASTER
    else:
        flags += gdal.OF_VECTOR

    gd = gdal.OpenEx(dso.path, flags, open_options=_option_list(dso.gdalOpts))
    if gd is None:
        raise Error(f'cannot open {dso.path!r}')

    if need_raster:
        return RasterDataSet(dso, gd)
    return VectorDataSet(dso, gd)


def open_from_image(
    image: gws.Image,
    bounds: gws.Bounds,
    rotation: gws.Size = None,
    options: dict = None,
) -> 'RasterDataSet':
    """Create an in-memory Dataset from an Image.

    Args:
        image: Image object.
        bounds: Geographic bounds.
        x_rotation: GeoTransform x rotation.
        y_rotation: GeoTransform y rotation.
        options: Driver-specific creation options.
    """

    gdal.UseExceptions()

    drv = gdal.GetDriverByName('MEM')
    img_array = image.to_array()
    band_count = img_array.shape[2]

    gd = drv.Create(
        '',
        xsize=img_array.shape[1],
        ysize=img_array.shape[0],
        bands=band_count,
        eType=gdal.GDT_Byte,
        options=_option_list(options),
    )
    for band in range(band_count):
        gd.GetRasterBand(band + 1).WriteArray(img_array[:, :, band])

    gt = _bounds_to_geotransform(bounds, (gd.RasterXSize, gd.RasterYSize), rotation)

    gd.SetGeoTransform(gt)
    gd.SetSpatialRef(_srs_from_srid(bounds.crs.srid))

    dso = _DataSetOptions(path='')
    return RasterDataSet(dso, gd)


##


class _DataSet:
    gdDataset: gdal.Dataset
    gdDriver: gdal.Driver
    dso: _DataSetOptions
    driverName: str

    def __init__(self, dso: _DataSetOptions, gd_dataset):
        self.gdDataset = gd_dataset
        self.gdDriver = self.gdDataset.GetDriver()
        self.driverName = self.gdDriver.GetDescription()
        self.dso = dso

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        self.gdDataset.FlushCache()
        setattr(self, 'gdDataset', None)

    def crs(self) -> Optional[gws.Crs]:
        srid = _srid_from_srs(self.gdDataset.GetSpatialRef())
        return gws.lib.crs.get(srid) if srid else None

    def set_crs(self, crs: gws.Crs):
        srs = _srs_from_srid(crs.srid)
        self.gdDataset.SetSpatialRef(srs)


class RasterDataSet(_DataSet):
    def to_image(self) -> gws.Image:
        """Convert the raster dataset to an Image object."""

        band_count = self.gdDataset.RasterCount
        x_size = self.gdDataset.RasterXSize
        y_size = self.gdDataset.RasterYSize

        arr_shape = (y_size, x_size, band_count)
        arr = np.zeros(arr_shape, dtype=np.uint8)

        for band in range(band_count):
            gd_band = self.gdDataset.GetRasterBand(band + 1)
            arr[:, :, band] = gd_band.ReadAsArray(0, 0, x_size, y_size)

        return gws.lib.image.from_array(arr)

    def create_copy(self, path: str, driver: str = '', strict=False, options: dict = None):
        """Create a copy of a DataSet.

        Args:
            path: Destination path.
            driver: Driver name, if omitted, will be suggested from the path extension.
            strict: If True, fail if some options are not supported.
            options: Driver-specific creation options.
        """

        gdal.UseExceptions()

        drv = _driver_from_args(path, driver, need_raster=True)
        gd = drv.CreateCopy(
            path,
            self.gdDataset,
            strict=1 if strict else 0,
            options=_option_list(options),
        )
        gd.SetMetadata(self.gdDataset.GetMetadata())
        gd.FlushCache()
        gd = None

    def bounds(self) -> gws.Bounds:
        return _geotransform_to_bounds(
            self.gdDataset.GetGeoTransform(),
            (self.gdDataset.RasterXSize, self.gdDataset.RasterYSize),
            self.crs() or self.dso.defaultCrs,
        )


class VectorDataSet(_DataSet):
    @contextlib.contextmanager
    def transaction(self):
        self.gdDataset.StartTransaction()
        try:
            yield self
            self.gdDataset.CommitTransaction()
        except:
            self.gdDataset.RollbackTransaction()
            raise

    def create_layer(
        self,
        name: str,
        columns: dict[str, gws.AttributeType],
        geometry_type: gws.GeometryType = None,
        crs: gws.Crs = None,
        overwrite=False,
        options: dict = None,
    ) -> 'VectorLayer':
        """Create a new layer.

        Args:
            name: Layer name.
            columns: Column definitions.
            geometry_type: Geometry type.
            crs: CRS for geometries.
            overwrite: If True, overwrite existing layer.
            options: Driver-specific creation options.
        """

        opts = dict(options or {})
        if overwrite:
            opts['OVERWRITE'] = 'YES'
        enc = (self.dso.encoding or '').upper()
        if enc:
            driver = self.gdDriver.GetName()
            if 'Shapefile' in driver:
                opts['ENCODING'] = enc

        geom_type = ogr.wkbUnknown
        srs = None

        if geometry_type:
            geom_type = _GEOM_TO_OGR.get(geometry_type)
            if not geom_type:
                gws.log.warning(f'gdal: unsupported {geometry_type=}')
                geom_type = ogr.wkbUnknown
            crs = crs or self.dso.defaultCrs
            srs = _srs_from_srid(crs.srid)

        gd_layer = self.gdDataset.CreateLayer(
            name,
            geom_type=geom_type,
            srs=srs,
            options=_option_list(opts),
        )
        for col_name, col_type in columns.items():
            gd_layer.CreateField(ogr.FieldDefn(col_name, _ATTR_TO_OGR[col_type]))

        return VectorLayer(self, gd_layer)

    def layers(self) -> list['VectorLayer']:
        """Get all layers."""

        cnt = self.gdDataset.GetLayerCount()
        return [VectorLayer(self, self.gdDataset.GetLayerByIndex(n)) for n in range(cnt)]

    def layer(self, name_or_index: str | int) -> Optional['VectorLayer']:
        """Get a layer by name or index."""

        gd_layer = None
        if isinstance(name_or_index, int):
            gd_layer = self.gdDataset.GetLayerByIndex(name_or_index)
        elif isinstance(name_or_index, str):
            gd_layer = self.gdDataset.GetLayerByName(name_or_index)
        return VectorLayer(self, gd_layer) if gd_layer else None

    def require_layer(self, name_or_index: str | int) -> 'VectorLayer':
        """Get a layer by name or index, raise an error if not found."""

        la = self.layer(name_or_index)
        if la:
            return la
        raise Error(f'layer {name_or_index} not found')


class VectorLayer:
    name: str
    dso: _DataSetOptions
    gdLayer: ogr.Layer
    gdDefn: ogr.FeatureDefn

    def __init__(self, ds: VectorDataSet, gd_layer: ogr.Layer):
        self.gdLayer = gd_layer
        self.gdDefn = self.gdLayer.GetLayerDefn()
        self.name = self.gdDefn.GetName()
        self.dso = ds.dso

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
            cols.append(
                gws.ColumnDescription(
                    name=fid_col,
                    type=_OGR_TO_ATTR[ogr.OFTInteger],
                    nativeType=ogr.OFTInteger,
                    isPrimaryKey=True,
                    columnIndex=0,
                )
            )

        for i in range(self.gdDefn.GetFieldCount()):
            fdef = self.gdDefn.GetFieldDefn(i)
            typ = fdef.GetType()
            if typ not in _OGR_TO_ATTR:
                continue
            cols.append(
                gws.ColumnDescription(
                    name=fdef.GetName(),
                    type=_OGR_TO_ATTR[typ],
                    nativeType=typ,
                    columnIndex=i,
                )
            )

        for i in range(self.gdDefn.GetGeomFieldCount()):
            fdef: ogr.GeomFieldDefn = self.gdDefn.GetGeomFieldDefn(i)
            typ = fdef.GetType()
            cols.append(
                gws.ColumnDescription(
                    name=fdef.GetName() or 'geom',
                    type=gws.AttributeType.geometry,
                    nativeType=typ,
                    columnIndex=i,
                    geometryType=_OGR_TO_GEOM.get(typ) or gws.GeometryType.geometry,
                    geometrySrid=_srid_from_srs(fdef.GetSpatialRef()) or self.dso.defaultCrs.srid,
                )
            )

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
                        _srs_from_srid(rec.shape.crs.srid),
                    )
                )

            if rec.uid and isinstance(rec.uid, int):
                gd_feature.SetFID(rec.uid)

            for col in desc.columns:
                if col.geometryType or col.isPrimaryKey:
                    continue
                val = rec.attributes.get(col.name)
                if val is None:
                    continue
                try:
                    _attr_to_ogr(gd_feature, int(col.nativeType), col.columnIndex, val, self.dso.encoding)
                except Exception as exc:
                    raise Error(f'field cannot be set: {col.name=} {val=}') from exc

            self.gdLayer.CreateFeature(gd_feature)
            fids.append(gd_feature.GetFID())

        return fids

    def count(self, force=False):
        return self.gdLayer.GetFeatureCount(force=1 if force else 0)

    def get_all(self) -> list[gws.FeatureRecord]:
        return list(self.iter_features())

    def iter_features(self) -> Iterable[gws.FeatureRecord]:
        self.gdLayer.ResetReading()

        while True:
            gd_feature = self.gdLayer.GetNextFeature()
            if not gd_feature:
                break
            yield self._feature_record(gd_feature)

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
            val = _attr_from_ogr(gd_feature, gd_field_defn.type, i, self.dso.encoding)
            rec.attributes[name] = val

        cnt = gd_feature.GetGeomFieldCount()
        if cnt > 0:
            # NB take the last geom
            # @TODO multigeometry support
            gd_geom_defn = gd_feature.GetGeomFieldRef(cnt - 1)
            if gd_geom_defn:
                srid = _srid_from_srs(gd_geom_defn.GetSpatialReference()) or self.dso.defaultCrs.srid
                wkt = gd_geom_defn.ExportToWkt()
                if self.dso.geometryAsText:
                    rec.ewkt = f'SRID={srid};{wkt}'
                else:
                    rec.shape = gws.base.shape.from_wkt(wkt, gws.lib.crs.get(srid))

        return rec


##


def _driver_from_args(path, driver_name, need_raster):
    di = _fetch_driver_infos()

    if not driver_name:
        ext = path.split('.')[-1]
        names = di.extToName.get(ext)
        if not names:
            raise Error(f'no default driver found for {path!r}')
        if len(names) > 1:
            if ext in ('tif', 'tiff'):
                driver_name = 'GTiff'
            else:
                raise Error(f'multiple drivers found for {path!r}: {names}')
        else:
            driver_name = names[0]

    is_vector = driver_name in di.vectorNames
    is_raster = driver_name in di.rasterNames

    if need_raster:
        if not is_raster:
            raise Error(f'driver {driver_name!r} is not raster')
        return gdal.GetDriverByName(driver_name)

    if not is_vector:
        raise Error(f'driver {driver_name!r} is not vector')
    return ogr.GetDriverByName(driver_name)


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
            metaData=dict(drv.GetMetadata() or {}),
        )
        _di_cache.infos.append(inf)

        for e in inf.metaData.get(gdal.DMD_EXTENSIONS, '').split():
            _di_cache.extToName.setdefault(e, []).append(inf.name)
        if inf.metaData.get('DCAP_VECTOR') == 'YES':
            _di_cache.vectorNames.add(inf.name)
        if inf.metaData.get('DCAP_RASTER') == 'YES':
            _di_cache.rasterNames.add(inf.name)

    return _di_cache


_name_to_srid = {}


def _srs_from_srid(srid):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(srid)
    return srs


def _srid_from_srs(srs):
    if not srs:
        return 0

    name = srs.GetName()
    if not name:
        wkt = srs.ExportToWkt()
        gws.log.warning(f'gdalx: no name for SRS {wkt!r}')
        return 0

    if name in _name_to_srid:
        return _name_to_srid[name]

    srid = srs.GetAuthorityCode(None)
    if not srid:
        wkt = srs.ExportToWkt()
        gws.log.warning(f'gdalx: no srid for SRS {wkt!r}')
        srid = 0

    _name_to_srid[name] = srid
    return srid


def _attr_from_ogr(gd_feature: ogr.Feature, gtype: int, idx: int, encoding: str = 'utf8'):
    if gd_feature.IsFieldNull(idx):
        return None

    if gtype == ogr.OFTString:
        b = gd_feature.GetFieldAsBinary(idx)
        if encoding:
            return b.decode(encoding)
        return bytes(b)

    if gtype in {ogr.OFTDate, ogr.OFTTime, ogr.OFTDateTime}:
        # python GetFieldAsDateTime appears to use float seconds, as in
        # GetFieldAsDateTime (int i, int *pnYear, int *pnMonth, int *pnDay, int *pnHour, int *pnMinute, float *pfSecond, int *pnTZFlag)
        #
        v = gd_feature.GetFieldAsDateTime(idx)
        sec, fsec = divmod(v[5], 1)
        try:
            return datetimex.new(v[0], v[1], v[2], v[3], v[4], int(sec), int(fsec * 1e6), tz=_tzflag_to_tz(v[6]))
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
        return 'UTC'
    if tzflag % 4 != 0:
        # @TODO
        raise Error(f'unsupported timezone {tzflag=}')
    hrs = (100 - tzflag) // 4
    return f'Etc/GMT{hrs:+}'


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


def is_attribute_supported(typ):
    return typ in _ATTR_TO_OGR


def _bounds_to_geotransform(bounds: gws.Bounds, px_size: gws.Size, rotation: gws.Size | None) -> tuple[float, float, float, float, float, float]:
    ext = bounds.extent
    res_x = (ext[2] - ext[0]) / px_size[0]
    res_y = (ext[1] - ext[3]) / px_size[1]
    xr = rotation[0] if rotation else 0.0
    yr = rotation[1] if rotation else 0.0
    return (ext[0], res_x, xr, ext[3], yr, res_y)


def _geotransform_to_bounds(gt: tuple[float, float, float, float, float, float], px_size: gws.Size, crs: gws.Crs) -> gws.Bounds:
    x0 = gt[0]
    x1 = x0 + gt[1] * px_size[0]
    y1 = gt[3]
    y0 = y1 + gt[5] * px_size[1]
    return gws.lib.bounds.from_extent((x0, y0, x1, y1), crs, always_xy=True)


def _option_list(opts: dict | None) -> list[str]:
    if not opts:
        return []
    return [f'{k}={v}' for k, v in opts.items()]


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
