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
            ds.create_copy(f'{d}/a.png')

        with gdalx.open_raster(f'{d}/a.png') as ds:
            ds.create_copy(f'{d}/b.tif')

        with gdalx.open_raster(f'{d}/a.png') as ds:
            img = ds.to_image()
            img.to_path(f'{d}/c.jpeg')

        # @TODO: verify content and transforms
