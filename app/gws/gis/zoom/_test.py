"""Tests for the zoom module."""

import pytest

import gws
import gws.gis.zoom as zoom
import gws.lib.crs
import gws.lib.grid


def _ladder(z):
    return gws.lib.grid.resolution_for_level(gws.lib.grid.for_crs(gws.lib.crs.WEBMERCATOR), z)


def test_map_default_is_ladder():
    res = zoom.resolutions_from_config(None, gws.lib.crs.WEBMERCATOR)
    assert len(res) == zoom.DEFAULT_MAX_LEVEL + 1
    assert res[-1] == _ladder(0)
    assert res[0] == _ladder(zoom.DEFAULT_MAX_LEVEL)


def test_map_levels():
    cnfg = zoom.Config(minLevel=5, maxLevel=15)
    res = zoom.resolutions_from_config(cnfg, gws.lib.crs.WEBMERCATOR)
    assert len(res) == 11
    assert res[-1] == _ladder(5)
    assert res[0] == _ladder(15)


def test_map_min_scale_extends_ladder():
    cnfg = zoom.Config(minScale=50)
    res = zoom.resolutions_from_config(cnfg, gws.lib.crs.WEBMERCATOR)
    assert res[0] == _ladder(23)


def test_map_min_resolution_snaps_to_ladder():
    cnfg = zoom.Config(minResolution=_ladder(22))
    res = zoom.resolutions_from_config(cnfg, gws.lib.crs.WEBMERCATOR)
    assert res[0] == _ladder(22)


def test_map_levels_and_scales_intersect():
    cnfg = zoom.Config(minLevel=5, maxScale=1_000_000)
    res = zoom.resolutions_from_config(cnfg, gws.lib.crs.WEBMERCATOR)
    assert res[-1] == _ladder(9)


def test_map_explicit_scales():
    cnfg = zoom.Config(scales=[1000, 5000, 25000])
    res = zoom.resolutions_from_config(cnfg, gws.lib.crs.WEBMERCATOR)
    assert [round(r / 0.00028) for r in res] == [1000, 5000, 25000]


def test_map_explicit_scales_with_max_level():
    cnfg = zoom.Config(scales=[1000, 5000, 25000], maxLevel=1)
    res = zoom.resolutions_from_config(cnfg, gws.lib.crs.WEBMERCATOR)
    assert [round(r / 0.00028) for r in res] == [5000, 25000]


def test_map_empty_range_raises():
    cnfg = zoom.Config(minLevel=10, maxLevel=5)
    with pytest.raises(gws.ConfigurationError):
        zoom.resolutions_from_config(cnfg, gws.lib.crs.WEBMERCATOR)


def test_layer_inherits_parent():
    parent = [_ladder(z) for z in range(0, 21)]
    assert zoom.resolutions_for_layer(None, parent) == sorted(parent)


def test_layer_scales_snap_to_parent():
    parent = [_ladder(z) for z in range(0, 21)]
    cnfg = zoom.Config(scales=[2000])
    res = zoom.resolutions_for_layer(cnfg, parent)
    assert res == [_ladder(18)]


def test_layer_levels_are_map_indices():
    parent = [_ladder(z) for z in range(0, 21)]
    cnfg = zoom.Config(minLevel=10, maxLevel=12)
    res = zoom.resolutions_for_layer(cnfg, parent)
    assert res == [_ladder(12), _ladder(11), _ladder(10)]


def test_layer_resolutions_must_match_parent():
    parent = [_ladder(z) for z in range(0, 21)]
    assert zoom.resolutions_for_layer(zoom.Config(resolutions=[_ladder(10)]), parent) == [_ladder(10)]
    with pytest.raises(gws.ConfigurationError):
        zoom.resolutions_for_layer(zoom.Config(resolutions=[123.0]), parent)


def test_init_level():
    res = [_ladder(z) for z in range(0, 21)]
    assert zoom.init_resolution(zoom.Config(initLevel=0), res) == _ladder(0)
    assert zoom.init_resolution(zoom.Config(initLevel=25), res) == _ladder(20)
    with pytest.raises(gws.ConfigurationError):
        zoom.init_resolution(zoom.Config(initLevel=99), res)


def test_limits():
    with pytest.raises(gws.ConfigurationError):
        zoom.resolutions_from_config(zoom.Config(minLevel=zoom.MAX_LEVEL + 1), gws.lib.crs.WEBMERCATOR)
    with pytest.raises(gws.ConfigurationError):
        zoom.resolutions_from_config(zoom.Config(scales=[0.5]), gws.lib.crs.WEBMERCATOR)
    with pytest.raises(gws.ConfigurationError):
        zoom.resolutions_from_config(zoom.Config(maxScale=2_000_000_000), gws.lib.crs.WEBMERCATOR)
    res = zoom.resolutions_from_config(zoom.Config(minScale=zoom.MIN_SCALE), gws.lib.crs.WEBMERCATOR)
    assert res[0] >= _ladder(zoom.MAX_LEVEL)


def test_init_scale_snaps():
    res = [_ladder(z) for z in range(0, 21)]
    assert zoom.init_resolution(zoom.Config(initScale=2000), res) == _ladder(18)


def test_init_default_is_middle():
    res = [_ladder(z) for z in range(0, 21)]
    assert zoom.init_resolution(None, res) == _ladder(10)


def test_source_layers_snap_to_parent():
    parent = [_ladder(z) for z in range(0, 21)]
    sl = gws.SourceLayer(scaleRange=[1000, 100000])
    res = zoom.resolutions_from_source_layers([sl], parent)
    assert res
    assert min(res) >= _ladder(20)
    assert max(res) <= _ladder(0)


def test_source_layers_no_scale_range():
    parent = [_ladder(z) for z in range(0, 21)]
    sls = [gws.SourceLayer(), gws.SourceLayer()]
    assert zoom.resolutions_from_source_layers(sls, parent) == parent


def test_source_layers_disjoint():
    sl = gws.SourceLayer(scaleRange=[100_000_000, 200_000_000])
    parent = [_ladder(20), _ladder(19)]
    assert zoom.resolutions_from_source_layers([sl], parent) == []
