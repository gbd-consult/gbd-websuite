"""Tests for GDAL raster data handling."""

import numpy as np
from osgeo import gdal

import gws
import gws.lib.gdalx as gdalx
import gws.lib.bounds
import gws.lib.crs
import gws.lib.image
import gws.lib.mime
import gws.test.util as u


def test_from_and_to_image(tmp_path):
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with gdalx.open_from_image(img, b) as ds:
        ds.save_as(f'{tmp_path}/a.png')

    with gdalx.open_raster(f'{tmp_path}/a.png') as ds:
        ds.save_as(f'{tmp_path}/b.tif')

    with gdalx.open_raster(f'{tmp_path}/a.png') as ds:
        img = ds.to_image()
        img.to_path(f'{tmp_path}/c.jpeg')

    # @TODO: verify content and transforms


def test_size():
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with gdalx.open_from_image(img, b) as ds:
        assert ds.size() == (100, 200)


def test_size_from_file(tmp_path):
    img = gws.lib.image.from_size((320, 240), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with gdalx.open_from_image(img, b) as ds:
        ds.save_as(f'{tmp_path}/sized.tif')

    with gdalx.open_raster(f'{tmp_path}/sized.tif') as ds:
        assert ds.size() == (320, 240)


def test_warp_to_path(tmp_path):
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with gdalx.open_from_image(img, b) as ds:
        ds.warp_to_path(f'{tmp_path}/resized.tif', {'width': 50, 'height': 100})
    with gdalx.open_raster(f'{tmp_path}/resized.tif') as ds:
        assert ds.size() == (50, 100)


def test_warp_to_path_upscale(tmp_path):
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with gdalx.open_from_image(img, b) as ds:
        ds.warp_to_path(f'{tmp_path}/resized.tif', {'width': 400, 'height': 800})
    with gdalx.open_raster(f'{tmp_path}/resized.tif') as ds:
        assert ds.size() == (400, 800)


def test_warp_to_path_save(tmp_path):
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with gdalx.open_from_image(img, b) as ds:
        ds.warp_to_path(f'{tmp_path}/resized.tif', {'width': 50, 'height': 100})

    with gdalx.open_raster(f'{tmp_path}/resized.tif') as ds:
        assert ds.size() == (50, 100)


def test_warp_to_path_to_image(tmp_path):
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with gdalx.open_from_image(img, b) as ds:
        ds.warp_to_path(f'{tmp_path}/resized.tif', {'width': 50, 'height': 100})
    with gdalx.open_raster(f'{tmp_path}/resized.tif') as ds:
        out_img = ds.to_image()
        assert out_img.size() == (50, 100)


def test_warp_to_path_algorithms(tmp_path):
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    for alg in ('near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average'):
        with gdalx.open_from_image(img, b) as ds:
            ds.warp_to_path(f'{tmp_path}/{alg}.tif', {'width': 50, 'height': 100, 'resampleAlg': alg})
        with gdalx.open_raster(f'{tmp_path}/{alg}.tif') as ds:
            assert ds.size() == (50, 100), f'warp_to_path with {alg} failed'


##


def _gradient_image(w, h):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 0] = np.arange(w, dtype=np.uint8)[None, :]
    arr[:, :, 1] = np.arange(h, dtype=np.uint8)[:, None]
    arr[:, :, 2] = 7
    arr[:, :, 3] = 255
    arr[: h // 2, : w // 2, 3] = 0
    return gws.lib.image.from_array(arr)


def _assert_extent(a, b):
    for x, y in zip(a, b):
        assert abs(x - y) < 1e-6


def test_from_bytes_png(tmp_path):
    img = _gradient_image(100, 200)
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)

    with gdalx.open_from_bytes(img.to_bytes(gws.lib.mime.PNG), b) as ds:
        assert ds.size() == (100, 200)
        ds.save_as(f'{tmp_path}/a.tif')

    with gdalx.open_raster(f'{tmp_path}/a.tif') as ds:
        assert ds.crs().srid == 3857
        _assert_extent(ds.bounds().extent, b.extent)
        assert np.array_equal(ds.to_image().to_array(), img.to_array())


def test_from_bytes_same_as_from_image(tmp_path):
    img = _gradient_image(100, 200)
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)

    with gdalx.open_from_image(img, b) as ds:
        ds.save_as(f'{tmp_path}/from_image.tif')
    with gdalx.open_from_bytes(img.to_bytes(gws.lib.mime.PNG), b) as ds:
        ds.save_as(f'{tmp_path}/from_bytes.tif')

    with gdalx.open_raster(f'{tmp_path}/from_image.tif') as ds_1, gdalx.open_raster(f'{tmp_path}/from_bytes.tif') as ds_2:
        assert ds_1.gdDataset.GetGeoTransform() == ds_2.gdDataset.GetGeoTransform()
        assert ds_1.crs().srid == ds_2.crs().srid
        assert np.array_equal(ds_1.to_image().to_array(), ds_2.to_image().to_array())


def test_from_bytes_paletted_png_is_expanded(tmp_path):
    img = _gradient_image(100, 200)
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    png8 = img.to_bytes(gws.lib.mime.PNG, {'mode': 'P'})
    expected = gws.lib.image.from_bytes(png8).convert('RGBA').to_array()

    with gdalx.open_from_bytes(png8, b) as ds:
        assert ds.gdDataset.RasterCount == 4
        ds.save_as(f'{tmp_path}/a.tif')

    with gdalx.open_raster(f'{tmp_path}/a.tif') as ds:
        assert ds.gdDataset.RasterCount == 4
        arr = ds.to_image().to_array()
        assert np.array_equal(arr, expected)
        assert arr[0, 0, 3] == 0
        assert arr[199, 99, 3] == 255


def test_from_bytes_jpeg(tmp_path):
    img = _gradient_image(100, 200)
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)

    with gdalx.open_from_bytes(img.to_bytes(gws.lib.mime.JPEG), b) as ds:
        assert ds.gdDataset.RasterCount == 3
        assert ds.size() == (100, 200)
        _assert_extent(ds.bounds().extent, b.extent)


def test_from_bytes_yx_crs(tmp_path):
    img = _gradient_image(100, 200)
    b = gws.lib.bounds.from_extent((7.0, 50.0, 8.0, 52.0), gws.lib.crs.WGS84, always_xy=True)

    with gdalx.open_from_bytes(img.to_bytes(gws.lib.mime.PNG), b) as ds:
        ds.save_as(f'{tmp_path}/a.tif')

    with gdalx.open_raster(f'{tmp_path}/a.tif') as ds:
        assert ds.crs().srid == 4326
        _assert_extent(ds.bounds().extent, b.extent)


def test_from_bytes_to_gpkg(tmp_path):
    img = _gradient_image(256, 256)
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6642000), gws.lib.crs.WEBMERCATOR)

    with gdalx.open_from_bytes(img.to_bytes(gws.lib.mime.PNG, {'mode': 'P'}), b) as ds:
        ds.save_as(f'{tmp_path}/a.gpkg')

    with gdalx.open_raster(f'{tmp_path}/a.gpkg') as ds:
        assert ds.size() == (256, 256)
        assert ds.crs().srid == 3857
        _assert_extent(ds.bounds().extent, b.extent)


def test_from_bytes_removes_memory_files():
    img = _gradient_image(100, 200)
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)

    with gdalx.open_from_bytes(img.to_bytes(gws.lib.mime.PNG), b) as ds:
        paths = ds.dso.memPaths
        assert len(paths) == 2
        for p in paths:
            assert gdal.VSIStatL(p) is not None

    for p in paths:
        assert gdal.VSIStatL(p) is None


def test_from_bytes_invalid_content():
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with u.raises(gdalx.Error):
        gdalx.open_from_bytes(b'bytes_1', b)
