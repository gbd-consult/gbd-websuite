"""Tests for the cache command-line parameters."""

import gws
import gws.gis.cache.cli as cli
import gws.test.util as u


def test_levels():
    assert cli._levels('') == []
    assert cli._levels('3') == [3]
    assert cli._levels('1,3,5') == [1, 3, 5]
    assert cli._levels('2-5') == [2, 3, 4, 5]


def test_filter():
    p = cli.FilterParams(layerUids=['layer_1'], cacheNames=['cache_1', 'cache_2'], crs=['3857'])
    flt = cli._filter(p)
    assert flt.layerUids == ['layer_1']
    assert flt.cacheNames == ['cache_1', 'cache_2']
    assert flt.srids == [3857]
    assert not flt.bbox


def test_filter_without_geometry():
    flt = cli._filter_with_geom(cli.FilterParamsWithGeom(cacheNames=['cache_1']))
    assert flt.cacheNames == ['cache_1']
    assert not flt.bbox


def test_filter_with_bbox():
    flt = cli._filter_with_geom(cli.FilterParamsWithGeom(crs=[25832], bbox=[1, 2, 3, 4]))
    assert flt.bbox.crs.srid == 25832
    assert flt.bbox.extent == (1, 2, 3, 4)


def test_filter_with_wkt_uses_its_envelope():
    flt = cli._filter_with_geom(cli.FilterParamsWithGeom(crs=[25832], wkt='POLYGON((1 2, 5 2, 3 8, 1 2))'))
    assert flt.bbox.crs.srid == 25832
    assert flt.bbox.extent == (1, 2, 5, 8)


def test_filter_with_geometry_requires_one_crs():
    with u.raises(gws.Error):
        cli._filter_with_geom(cli.FilterParamsWithGeom(bbox=[1, 2, 3, 4]))
    with u.raises(gws.Error):
        cli._filter_with_geom(cli.FilterParamsWithGeom(crs=[3857, 25832], bbox=[1, 2, 3, 4]))


def test_filter_with_bbox_and_wkt_fails():
    with u.raises(gws.Error):
        cli._filter_with_geom(cli.FilterParamsWithGeom(crs=[3857], bbox=[1, 2, 3, 4], wkt='POINT(1 2)'))


def test_filter_with_invalid_bbox_fails():
    with u.raises(gws.Error):
        cli._filter_with_geom(cli.FilterParamsWithGeom(crs=[3857], bbox=[1, 2, 3]))
