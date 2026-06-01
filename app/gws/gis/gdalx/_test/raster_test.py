"""Tests for GDAL raster data handling."""


import gws
import gws.lib.gdalx as gdalx
import gws.lib.bounds
import gws.lib.crs
import gws.lib.image
import gws.test.util as u


def test_from_and_to_image():
    with u.temp_dir_in_base_dir(True) as d:
        img = gws.lib.image.from_size((100, 200), '#ff00ff32')
        b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
        with gdalx.open_from_image(img, b) as ds:
            ds.save_as(f'{d}/a.png')

        with gdalx.open_raster(f'{d}/a.png') as ds:
            ds.save_as(f'{d}/b.tif')

        with gdalx.open_raster(f'{d}/a.png') as ds:
            img = ds.to_image()
            img.to_path(f'{d}/c.jpeg')

        # @TODO: verify content and transforms


def test_size():
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with gdalx.open_from_image(img, b) as ds:
        assert ds.size() == (100, 200)


def test_size_from_file():
    with u.temp_dir_in_base_dir(True) as d:
        img = gws.lib.image.from_size((320, 240), '#ff00ff32')
        b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
        with gdalx.open_from_image(img, b) as ds:
            ds.save_as(f'{d}/sized.tif')

        with gdalx.open_raster(f'{d}/sized.tif') as ds:
            assert ds.size() == (320, 240)


def test_warp_to_path():
    with u.temp_dir_in_base_dir(True) as d:
        img = gws.lib.image.from_size((100, 200), '#ff00ff32')
        b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
        with gdalx.open_from_image(img, b) as ds:
            ds.warp_to_path(f'{d}/resized.tif', {'width': 50, 'height': 100})
        with gdalx.open_raster(f'{d}/resized.tif') as ds:
            assert ds.size() == (50, 100)


def test_warp_to_path_upscale():
    with u.temp_dir_in_base_dir(True) as d:
        img = gws.lib.image.from_size((100, 200), '#ff00ff32')
        b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
        with gdalx.open_from_image(img, b) as ds:
            ds.warp_to_path(f'{d}/resized.tif', {'width': 400, 'height': 800})
        with gdalx.open_raster(f'{d}/resized.tif') as ds:
            assert ds.size() == (400, 800)


def test_warp_to_path_save():
    with u.temp_dir_in_base_dir(True) as d:
        img = gws.lib.image.from_size((100, 200), '#ff00ff32')
        b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
        with gdalx.open_from_image(img, b) as ds:
            ds.warp_to_path(f'{d}/resized.tif', {'width': 50, 'height': 100})

        with gdalx.open_raster(f'{d}/resized.tif') as ds:
            assert ds.size() == (50, 100)


def test_warp_to_path_to_image():
    with u.temp_dir_in_base_dir(True) as d:
        img = gws.lib.image.from_size((100, 200), '#ff00ff32')
        b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
        with gdalx.open_from_image(img, b) as ds:
            ds.warp_to_path(f'{d}/resized.tif', {'width': 50, 'height': 100})
        with gdalx.open_raster(f'{d}/resized.tif') as ds:
            out_img = ds.to_image()
            assert out_img.size() == (50, 100)


def test_warp_to_path_algorithms():
    with u.temp_dir_in_base_dir(True) as d:
        img = gws.lib.image.from_size((100, 200), '#ff00ff32')
        b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
        for alg in ('near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average'):
            with gdalx.open_from_image(img, b) as ds:
                ds.warp_to_path(f'{d}/{alg}.tif', {'width': 50, 'height': 100, 'resampleAlg': alg})
            with gdalx.open_raster(f'{d}/{alg}.tif') as ds:
                assert ds.size() == (50, 100), f'warp_to_path with {alg} failed'
