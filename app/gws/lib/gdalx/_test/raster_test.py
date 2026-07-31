"""Tests for GDAL raster data handling."""


import gws
import gws.lib.gdalx as gdalx
import gws.lib.bounds
import gws.lib.crs
import gws.lib.image
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
