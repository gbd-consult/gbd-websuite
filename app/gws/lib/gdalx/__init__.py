"""GDAL/OGR wrapper.

This package provides a thin layer over the GDAL Python bindings (``osgeo.gdal``, ``osgeo.ogr``)
for reading and writing raster and vector data sets.

Data sets are opened with ``open_raster`` or ``open_vector``, or created in memory
from an image with ``open_from_image``. If no driver name is given, the driver is chosen
by the file extension. Data sets are context managers that flush and close themselves on exit.

- ``RasterDataSet`` reads a raster into a ``gws.Image``, reports its size and bounds,
  warps it to an image or a file (``gdal.Warp``), and saves copies in other formats.
- ``VectorDataSet`` gives access to ``VectorLayer`` objects, creates new layers and runs transactions.
- ``VectorLayer`` describes its columns, reads features as ``gws.FeatureRecord`` objects and inserts records.

Attribute values are converted between OGR field types and ``gws.AttributeType``,
geometries between OGR and ``gws.Shape``. Geo-transforms always use the
easting/longitude-first axis order.

Example::

    import gws.lib.gdalx

    with gws.lib.gdalx.open_raster('/data/ortho.tif') as ds:
        bounds = ds.bounds()
        img = ds.to_image()

    with gws.lib.gdalx.open_vector('/data/out.gpkg', 'w') as ds:
        la = ds.create_layer('roads', {'name': gws.AttributeType.str}, gws.GeometryType.linestring, crs)
        la.insert(records)
"""

from typing import Any, Optional, Iterable, cast

import datetime
import decimal
import contextlib
import numpy as np

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import gws
import gws.lib.shape
import gws.lib.crs
import gws.lib.bounds
import gws.lib.image
import gws.lib.datetimex as datetimex


class Error(gws.Error):
    """GDAL error."""

    pass


class DriverInfo(gws.Data):
    """Information about a GDAL driver."""

    index: int
    """Driver index in GDAL."""
    name: str
    """Short driver name, like ``GTiff``."""
    longName: str
    """Long driver name, like ``GeoTIFF``."""
    extensions: list[str]
    """File extensions supported by the driver."""
    metaData: dict
    """Driver metadata."""


def get_drivers() -> list[DriverInfo]:
    """Enumerate GDAL drivers.

    Returns:
        Information about all available drivers.
    """

    di = gws.u.get_app_global('gdal_driver_infos', _fetch_driver_infos)
    return di.infos


def get_driver(name: str) -> Optional[DriverInfo]:
    """Get driver info by name.

    Args:
        name: Short driver name, like ``GTiff``.

    Returns:
        Driver information, or ``None`` if the driver is not found.
    """

    for di in get_drivers():
        if di.name == name:
            return di


def supported_attribute_types():
    """Get attribute types that can be written to vector data sets.

    Returns:
        A list of ``gws.AttributeType`` values.
    """
    return list(_ATTR_TO_OGR.keys())


@contextlib.contextmanager
def gdal_config(options: dict):
    """Context manager that temporarily sets GDAL config options.

    The previous values are restored on exit.

    Args:
        options: GDAL config options, like ``{'GDAL_CACHEMAX': '512'}``.
    """

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
    """Open a raster data set.

    Args:
        path: File path.
        mode: ``r`` (read), ``a`` (update) or ``w`` (create).
        driver: Driver name. If omitted, the driver is chosen by the path extension.
        default_crs: CRS to use if the data set has none, web mercator by default.
        options: Driver-specific open or creation options.

    Returns:
        The raster data set.

    Raises:
        ``Error``: If the mode is invalid, no suitable raster driver is found, or the data set cannot be opened or created.
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
    """Open a vector data set.

    Args:
        path: File path.
        mode: ``r`` (read), ``a`` (update) or ``w`` (create).
        driver: Driver name. If omitted, the driver is chosen by the path extension.
        encoding: Encoding of string attributes. If set, strings are decoded when reading,
            otherwise they are returned as bytes.
        default_crs: CRS for geometries without one, web mercator by default.
        geometry_as_text: Do not convert geometries to shapes, return them as EWKT in ``FeatureRecord.ewkt``.
        options: Driver-specific open or creation options.

    Returns:
        The vector data set.

    Raises:
        ``Error``: If the mode is invalid, no suitable vector driver is found, or the data set cannot be opened or created.
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


def open_from_image(
    image: gws.Image,
    bounds: gws.Bounds,
    rotation: gws.Size = None,
    options: dict = None,
) -> 'RasterDataSet':
    """Create an in-memory raster data set from an image.

    Args:
        image: Image object.
        bounds: Bounds of the image.
        rotation: Geo-transform rotation terms ``(x, y)``, no rotation by default.
        options: Driver-specific creation options.

    Returns:
        The raster data set.
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


class _DriverInfoCache(gws.Data):
    """Cached information about GDAL drivers."""

    infos: list[DriverInfo]
    """All drivers."""
    extToName: dict
    """Map of file extensions to lists of driver names."""
    vectorNames: set[str]
    """Names of vector drivers."""
    rasterNames: set[str]
    """Names of raster drivers."""


class _DataSetOptions(gws.Data):
    """Options a data set was opened with."""

    path: str
    """File path."""
    mode: str
    """Open mode."""
    driver: str
    """Driver name."""
    encoding: str
    """Encoding of string attributes."""
    defaultCrs: gws.Crs
    """CRS to use if the data has none."""
    geometryAsText: bool
    """Return geometries as EWKT instead of shapes."""
    gdalOpts: dict
    """Driver-specific options."""


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


class _DataSet:
    """Base class for GDAL data sets."""

    gdDataset: gdal.Dataset
    """Underlying GDAL data set."""
    gdDriver: gdal.Driver
    """Underlying GDAL driver."""
    dso: _DataSetOptions
    """Options the data set was opened with."""
    driverName: str
    """Driver name."""

    def __init__(self, dso: _DataSetOptions, gd_dataset):
        """Wrap a GDAL data set.

        Args:
            dso: Options the data set was opened with.
            gd_dataset: GDAL data set.
        """
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
        """Flush the data set and release it."""
        self.gdDataset.FlushCache()
        setattr(self, 'gdDataset', None)

    def crs(self) -> Optional[gws.Crs]:
        """Get the CRS of the data set.

        Returns:
            The CRS, or ``None`` if the data set has no CRS or it is unknown.
        """
        srid = _srid_from_srs(self.gdDataset.GetSpatialRef())
        return gws.lib.crs.get(srid) if srid else None

    def set_crs(self, crs: gws.Crs):
        """Set the CRS of the data set.

        Args:
            crs: The CRS.
        """
        srs = _srs_from_srid(crs.srid)
        self.gdDataset.SetSpatialRef(srs)


class RasterDataSet(_DataSet):
    """Raster data set."""

    def to_image(self) -> gws.Image:
        """Convert the raster data set to an image.

        Each raster band becomes an image channel, values are read as 8-bit.

        Returns:
            The image.
        """

        band_count = self.gdDataset.RasterCount
        x_size = self.gdDataset.RasterXSize
        y_size = self.gdDataset.RasterYSize

        arr_shape = (y_size, x_size, band_count)
        arr = np.zeros(arr_shape, dtype=np.uint8)

        for band in range(band_count):
            gd_band = self.gdDataset.GetRasterBand(band + 1)
            arr[:, :, band] = gd_band.ReadAsArray(0, 0, x_size, y_size)

        return gws.lib.image.from_array(arr)

    def warp_to_image(self, options: dict) -> gws.Image:
        """Warp the data set in memory and return the result as an image.

        See https://gdal.org/en/stable/api/python/utilities.html#osgeo.gdal.WarpOptions
        and https://gdal.org/en/stable/programs/gdalwarp.html for the options.

        Args:
            options: Keyword arguments for ``gdal.Warp``. The ``format`` option is ignored.

        Returns:
            The warped image.

        Raises:
            ``Error``: If the warp fails.
        """

        gdal.UseExceptions()

        options = dict(options)
        options['format'] = 'MEM'

        gd = gdal.Warp('', self.gdDataset, **options)
        if gd is None:
            raise Error(f'warp failed')

        return RasterDataSet(_DataSetOptions(path=''), gd).to_image()

    def warp_to_path(self, path: str, options: dict):
        """Warp the data set and store it at the given path.

        See https://gdal.org/en/stable/api/python/utilities.html#osgeo.gdal.WarpOptions
        and https://gdal.org/en/stable/programs/gdalwarp.html for the options.

        Args:
            path: Destination path.
            options: Keyword arguments for ``gdal.Warp``. If ``format`` is not given, it is chosen by the path extension.

        Raises:
            ``Error``: If no driver is found for the path or the warp fails.
        """

        gdal.UseExceptions()

        if 'format' not in options:
            options = dict(options)
            options['format'] = _driver_from_args(path, '', True).GetDescription()

        gd = gdal.Warp(path, self.gdDataset, **options)
        if gd is None:
            raise Error(f'warp failed')
        gd.FlushCache()
        gd = None

    def save_as(self, path: str, driver: str = '', strict=False, options: dict = None):
        """Save a copy of the data set, including its metadata.

        Args:
            path: Destination path.
            driver: Driver name. If omitted, the driver is chosen by the path extension.
            strict: Fail if the copy cannot be made exactly, for example, if the format does not support some data.
            options: Driver-specific creation options.

        Raises:
            ``Error``: If no suitable raster driver is found.
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

    def size(self) -> gws.Size:
        """Get the raster size.

        Returns:
            A ``(width, height)`` tuple in pixels.
        """
        return (self.gdDataset.RasterXSize, self.gdDataset.RasterYSize)

    def bounds(self) -> gws.Bounds:
        """Get the bounds of the raster, computed from its geo-transform.

        Returns:
            The bounds, in the data set CRS or the default CRS.
        """
        return _geotransform_to_bounds(
            self.gdDataset.GetGeoTransform(),
            (self.gdDataset.RasterXSize, self.gdDataset.RasterYSize),
            self.crs() or self.dso.defaultCrs,
        )


class VectorDataSet(_DataSet):
    """Vector data set."""

    @contextlib.contextmanager
    def transaction(self):
        """Context manager that runs a transaction.

        The transaction is committed on success and rolled back on an exception.

        Yields:
            This data set.
        """
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

        For Shapefiles, the data set encoding is passed to the driver.

        Args:
            name: Layer name.
            columns: Map of column names to attribute types.
            geometry_type: Geometry type. If omitted, the layer has no geometry.
            crs: CRS for geometries, the default CRS of the data set by default.
            overwrite: Overwrite an existing layer.
            options: Driver-specific layer creation options.

        Returns:
            The new layer.
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
            fd = ogr.FieldDefn(col_name, _ATTR_TO_OGR[col_type])
            if col_type == gws.AttributeType.bool:
                fd.SetSubType(ogr.OFSTBoolean)
            gd_layer.CreateField(fd)

        return VectorLayer(self, gd_layer)

    def layers(self) -> list['VectorLayer']:
        """Get all layers.

        Returns:
            A list of layers.
        """

        cnt = self.gdDataset.GetLayerCount()
        return [VectorLayer(self, self.gdDataset.GetLayerByIndex(n)) for n in range(cnt)]

    def layer(self, name_or_index: str | int) -> Optional['VectorLayer']:
        """Get a layer by name or index.

        Args:
            name_or_index: Layer name or index.

        Returns:
            The layer, or ``None`` if not found.
        """

        gd_layer = None
        if isinstance(name_or_index, int):
            gd_layer = self.gdDataset.GetLayerByIndex(name_or_index)
        elif isinstance(name_or_index, str):
            gd_layer = self.gdDataset.GetLayerByName(name_or_index)
        return VectorLayer(self, gd_layer) if gd_layer else None

    def require_layer(self, name_or_index: str | int) -> 'VectorLayer':
        """Get a layer by name or index, and fail if it is not found.

        Args:
            name_or_index: Layer name or index.

        Returns:
            The layer.

        Raises:
            ``Error``: If the layer is not found.
        """

        la = self.layer(name_or_index)
        if la:
            return la
        raise Error(f'layer {name_or_index} not found')


class VectorLayer:
    """Layer of a vector data set."""

    name: str
    """Layer name."""
    dso: _DataSetOptions
    """Options of the data set."""
    gdLayer: ogr.Layer
    """Underlying OGR layer."""
    gdDefn: ogr.FeatureDefn
    """Underlying OGR feature definition."""

    def __init__(self, ds: VectorDataSet, gd_layer: ogr.Layer):
        """Wrap an OGR layer.

        Args:
            ds: Data set the layer belongs to.
            gd_layer: OGR layer.
        """
        self.gdLayer = gd_layer
        self.gdDefn = self.gdLayer.GetLayerDefn()
        self.name = self.gdDefn.GetName()
        self.dso = ds.dso

    def describe(self) -> gws.DataSetDescription:
        """Describe the layer columns.

        The description includes the FID column (as the primary key), attribute columns
        of supported types, and geometry columns. If there are several geometry columns,
        the last one is used as the layer geometry.

        Returns:
            The layer description.
        """
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
            attr_type = _OGR_TO_ATTR[typ]
            if fdef.GetSubType() == ogr.OFSTBoolean:
                attr_type = gws.AttributeType.bool
            cols.append(
                gws.ColumnDescription(
                    name=fdef.GetName(),
                    type=attr_type,
                    nativeType=typ,
                    columnIndex=i,
                )
            )

        for i in range(self.gdDefn.GetGeomFieldCount()):
            fdef = self.gdDefn.GetGeomFieldDefn(i)
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
        """Insert feature records into the layer.

        Integer record uids are used as feature ids. Attributes that are ``None``
        or have no matching column are skipped.

        Args:
            records: Feature records.

        Returns:
            Feature ids of the inserted features.

        Raises:
            ``Error``: If an attribute value cannot be set.
        """
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
        """Count features in the layer.

        Args:
            force: Count features even if this is expensive for the driver.

        Returns:
            The number of features, or ``-1`` if the count is not available without ``force``.
        """
        return self.gdLayer.GetFeatureCount(force=1 if force else 0)

    def get_all(self) -> list[gws.FeatureRecord]:
        """Read all features.

        Returns:
            A list of feature records.
        """
        return list(self.iter_features())

    def iter_features(self) -> Iterable[gws.FeatureRecord]:
        """Iterate over all features.

        Yields:
            Feature records. The record uid is the feature id as a string,
            ``meta['layerName']`` is the layer name.
        """
        self.gdLayer.ResetReading()

        while True:
            gd_feature = self.gdLayer.GetNextFeature()
            if not gd_feature:
                break
            yield self._feature_record(gd_feature)

    def get(self, fid: int) -> Optional[gws.FeatureRecord]:
        """Read a feature by its id.

        Args:
            fid: Feature id.

        Returns:
            The feature record, or ``None`` if not found.
        """
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
            fdef = gd_feature.GetFieldDefnRef(i)
            val = _attr_from_ogr(gd_feature, fdef.GetType(), fdef.GetSubType(), i, self.dso.encoding)
            rec.attributes[fdef.GetName()] = val

        cnt = gd_feature.GetGeomFieldCount()
        if cnt > 0:
            # NB take the last geom
            # @TODO multigeometry support
            fdef = gd_feature.GetGeomFieldRef(cnt - 1)
            if fdef:
                srid = _srid_from_srs(fdef.GetSpatialReference()) or self.dso.defaultCrs.srid
                if self.dso.geometryAsText:
                    rec.ewkt = f'SRID={srid};{fdef.ExportToWkt()}'
                else:
                    rec.shape = gws.lib.shape.from_wkb(bytes(fdef.ExportToIsoWkb()), gws.lib.crs.get(srid))

        return rec


##


def _driver_from_args(path, driver_name, need_raster):
    di = gws.u.get_app_global('gdal_driver_infos', _fetch_driver_infos)

    if not driver_name:
        ext = path.split('.')[-1]
        names = di.extToName.get(ext)
        if not names:
            raise Error(f'no default driver found for {path!r}')
        if len(names) == 1:
            driver_name = names[0]
        elif ext in _DEFAULT_DRIVERS:
            driver_name = _DEFAULT_DRIVERS[ext]
        else:
            raise Error(f'multiple drivers found for {path!r}: {names}')

    is_vector = driver_name in di.vectorNames
    is_raster = driver_name in di.rasterNames

    if need_raster:
        if not is_raster:
            raise Error(f'driver {driver_name!r} is not raster')
        return gdal.GetDriverByName(driver_name)

    if not is_vector:
        raise Error(f'driver {driver_name!r} is not vector')
    return ogr.GetDriverByName(driver_name)


_DEFAULT_DRIVERS = {
    'gif': 'GIF',
    'gml': 'GML',
    'kml': 'KML',
    'tif': 'GTiff',
    'tiff': 'GTiff',
}


def _fetch_driver_infos() -> _DriverInfoCache:
    dic = _DriverInfoCache(
        infos=[],
        extToName={},
        vectorNames=set(),
        rasterNames=set(),
    )

    for n in range(gdal.GetDriverCount()):
        drv = gdal.GetDriver(n)
        di = DriverInfo(
            index=n,
            name=str(drv.ShortName),
            longName=str(drv.LongName),
            extensions=[],
            metaData=dict(drv.GetMetadata() or {}),
        )
        dic.infos.append(di)

        for e in di.metaData.get(gdal.DMD_EXTENSIONS, '').split():
            dic.extToName.setdefault(e, []).append(di.name)
            di.extensions.append(e)
        if di.metaData.get('DCAP_VECTOR') == 'YES':
            dic.vectorNames.add(di.name)
        if di.metaData.get('DCAP_RASTER') == 'YES':
            dic.rasterNames.add(di.name)

    return dic


_name_to_srid = {}


def _srs_from_srid(srid):
    # our geotransforms are always x=easting/longitude, so the SRS must use the
    # traditional axis order, not the authority order (northing/latitude first
    # for e.g. 4326, 3035, 31466-31469, 3044-3045).

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(srid)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
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


def _attr_from_ogr(gd_feature: ogr.Feature, gtype: int, gsubtype: int, idx: int, encoding: str):
    if gd_feature.IsFieldNull(idx):
        return None

    if gtype == ogr.OFTString:
        b = gd_feature.GetFieldAsBinary(idx)
        if encoding:
            return b.decode(encoding)
        return bytes(b)

    # GetFieldAsDateTime uses float seconds:
    # GetFieldAsDateTime(int i, int *pnYear, int *pnMonth, int *pnDay, int *pnHour, int *pnMinute, float *pfSecond, int *pnTZFlag)

    if gtype == ogr.OFTDate:
        v = gd_feature.GetFieldAsDateTime(idx)
        return datetime.date(v[0], v[1], v[2])

    if gtype == ogr.OFTTime:
        v = gd_feature.GetFieldAsDateTime(idx)
        sec, fsec = divmod(v[5], 1)
        return datetime.time(v[3], v[4], int(sec), int(fsec * 1e6))

    if gtype == ogr.OFTDateTime:
        v = gd_feature.GetFieldAsDateTime(idx)
        sec, fsec = divmod(v[5], 1)
        return datetimex.new(v[0], v[1], v[2], v[3], v[4], int(sec), int(fsec * 1e6), tz=_tzflag_to_tz(v[6]))

    if gtype in {ogr.OFTIntegerList, ogr.OFTInteger64List}:
        return gd_feature.GetFieldAsIntegerList(idx)
    if gtype == ogr.OFTRealList:
        return gd_feature.GetFieldAsDoubleList(idx)
    if gtype == ogr.OFTStringList:
        return list(gd_feature.GetFieldAsStringList(idx))
    if gtype in {ogr.OFTInteger, ogr.OFTInteger64}:
        if gsubtype == ogr.OFSTBoolean:
            return gd_feature.GetFieldAsInteger(idx) != 0
        return gd_feature.GetFieldAsInteger(idx)
    if gtype == ogr.OFTReal:
        return gd_feature.GetFieldAsDouble(idx)
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


def _attr_to_ogr(gd_feature: ogr.Feature, gtype: int, idx: int, value: Any, encoding):
    if isinstance(value, decimal.Decimal):
        value = float(value)

    if gtype == ogr.OFTDate:
        return gd_feature.SetField(idx, datetimex.to_iso_date_string(value))
    if gtype == ogr.OFTTime:
        return gd_feature.SetField(idx, datetimex.to_iso_time_string(value))
    if gtype == ogr.OFTDateTime:
        return gd_feature.SetField(idx, datetimex.to_iso_string(datetimex.to_utc(value), with_tz='Z'))
    if gtype in {ogr.OFTInteger, ogr.OFTInteger64}:
        return gd_feature.SetField(idx, int(bool(value) if isinstance(value, bool) else value))
    if gtype in {ogr.OFTIntegerList, ogr.OFTInteger64List}:
        return gd_feature.SetFieldIntegerList(idx, [int(x) for x in value])
    if gtype == ogr.OFTRealList:
        return gd_feature.SetFieldDoubleList(idx, [float(x) for x in value])
    if gtype == ogr.OFTReal:
        return gd_feature.SetField(idx, float(value))
    if gtype == ogr.OFTString:
        if isinstance(value, bytes):
            return gd_feature.SetField(idx, value.decode(encoding or 'utf8'))
        return gd_feature.SetField(idx, str(value))
    if gtype == ogr.OFTStringList:
        return gd_feature.SetFieldStringList(idx, [str(x) for x in value])
    if gtype == ogr.OFTBinary:
        return gd_feature.SetFieldBinaryFromHexString(idx, value.hex() if isinstance(value, bytes) else value)

    return gd_feature.SetField(idx, value)


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
