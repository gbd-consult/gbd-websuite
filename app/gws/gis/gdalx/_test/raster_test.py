"""Tests for GDAL raster data handling."""


import gws
import gws.gis.gdalx as gdalx
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


def test_resize():
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with gdalx.open_from_image(img, b) as ds:
        resized = ds.resize((50, 100))
        assert resized.size() == (50, 100)
        resized.close()


def test_resize_upscale():
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with gdalx.open_from_image(img, b) as ds:
        resized = ds.resize((400, 800))
        assert resized.size() == (400, 800)
        resized.close()


def test_resize_save():
    with u.temp_dir_in_base_dir(True) as d:
        img = gws.lib.image.from_size((100, 200), '#ff00ff32')
        b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
        with gdalx.open_from_image(img, b) as ds:
            resized = ds.resize((50, 100))
            resized.save_as(f'{d}/resized.tif')
            resized.close()

        with gdalx.open_raster(f'{d}/resized.tif') as ds:
            assert ds.size() == (50, 100)


def test_resize_to_image():
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    with gdalx.open_from_image(img, b) as ds:
        resized = ds.resize((50, 100))
        out_img = resized.to_image()
        assert out_img.size() == (50, 100)
        resized.close()


def test_resize_algorithms():
    img = gws.lib.image.from_size((100, 200), '#ff00ff32')
    b = gws.lib.bounds.from_extent((753000, 6640000, 755000, 6641000), gws.lib.crs.WEBMERCATOR)
    for alg in ('NearestNeighbour', 'Bilinear', 'Cubic', 'Lanczos', 'Average'):
        with gdalx.open_from_image(img, b) as ds:
            resized = ds.resize((50, 100), alg=alg)
            assert resized.size() == (50, 100), f'resize with {alg} failed'
            resized.close()
