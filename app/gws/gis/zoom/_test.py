"""Tests for the zoom module."""

import gws
import gws.test.util as u
import gws.gis.zoom as zoom
import gws.lib.crs


def test_resolutions_from_config_empty_cnfg():
    cnfg = zoom.Config()
    assert zoom.resolutions_from_config(cnfg) == zoom.OSM_RESOLUTIONS


def test_resolutions_from_config():
    resolutions = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    minresolution = 3.0
    maxresolution = 4.0
    minscale = 1.0
    maxscale = 2.0
    scales = [1, 2, 3, 4]
    cnfg = zoom.Config(resolutions=resolutions, minResolution=minresolution, maxResolution=maxresolution,
                       minScale=minscale, maxScale=maxscale, scales=scales)
    assert zoom.resolutions_from_config(cnfg) == [3, 3.5, 4]


def test_resolutions_from_config_no_minmax():
    resolutions = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    scales = [1, 2, 3, 4]
    cnfg = zoom.Config(resolutions=resolutions, scales=scales)
    assert zoom.resolutions_from_config(cnfg) == [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]


def test_resolutions_from_config_parent():
    parentresolutions = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    cnfg = zoom.Config(minResolution=2.0, maxResolution=5.0)
    assert zoom.resolutions_from_config(cnfg, parentresolutions) == [2.0, 3.0, 4.0, 5.0]


def test_resolutions_from_source_layers():
    sl1 = gws.SourceLayer(scaleRange=[100, 100])
    sl2 = gws.SourceLayer(scaleRange=[110, 200])
    sl3 = gws.SourceLayer(scaleRange=[210, 30000])
    parentresolutions = [0.0, 0.01, 0.02, 5.0, 6.0, 7.0, 9.0, 10.0, 10000.0]
    assert zoom.resolutions_from_source_layers([sl3, sl2, sl1], parentresolutions) == [0.02, 5.0, 6.0, 7.0, 9.0]


def test_resolutions_from_source_layers_no_scalerange():
    sl1 = gws.SourceLayer()
    sl2 = gws.SourceLayer()
    sl3 = gws.SourceLayer()
    parentresolutions = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert zoom.resolutions_from_source_layers([sl3, sl2, sl1], parentresolutions) == parentresolutions


def test_resolutions_from_source_layers_empty():
    sl1 = gws.SourceLayer(scaleRange=[0, 10])
    parentresolutions = [1.0, 6.0]
    assert zoom.resolutions_from_source_layers([sl1], parentresolutions) == []


def test_resolutions_from_bounds():
    crs = gws.lib.crs.WGS84
    extent = (200, 200, 400, 400)
    bounds = gws.Bounds(crs=crs, extent=extent)
    assert zoom.resolutions_from_bounds(bounds, 3) == [66.66666666666667,
                                                       33.333333333333336,
                                                       16.666666666666668,
                                                       8.333333333333334,
                                                       4.166666666666667,
                                                       2.0833333333333335,
                                                       1.0416666666666667,
                                                       0.5208333333333334,
                                                       0.2604166666666667,
                                                       0.13020833333333334,
                                                       0.06510416666666667,
                                                       0.032552083333333336,
                                                       0.016276041666666668,
                                                       0.008138020833333334,
                                                       0.004069010416666667,
                                                       0.0020345052083333335,
                                                       0.0010172526041666667,
                                                       0.0005086263020833334,
                                                       0.0002543131510416667,
                                                       0.00012715657552083334]


def test_resolutions_from_bounds_zero():
    crs = gws.lib.crs.WGS84
    extent = (200, 200, 400, 400)
    bounds = gws.Bounds(crs=crs, extent=extent)
    with u.raises(Exception):
        zoom.resolutions_from_bounds(bounds, 0)


def test_resolutions_from_bounds_negative():
    crs = gws.lib.crs.WGS84
    extent = (200, 200, 400, 400)
    bounds = gws.Bounds(crs=crs, extent=extent)
    assert zoom.resolutions_from_bounds(bounds, -3) == [-66.66666666666667,
                                                        -33.333333333333336,
                                                        -16.666666666666668,
                                                        -8.333333333333334,
                                                        -4.166666666666667,
                                                        -2.0833333333333335,
                                                        -1.0416666666666667,
                                                        -0.5208333333333334,
                                                        -0.2604166666666667,
                                                        -0.13020833333333334,
                                                        -0.06510416666666667,
                                                        -0.032552083333333336,
                                                        -0.016276041666666668,
                                                        -0.008138020833333334,
                                                        -0.004069010416666667,
                                                        -0.0020345052083333335,
                                                        -0.0010172526041666667,
                                                        -0.0005086263020833334,
                                                        -0.0002543131510416667,
                                                        -0.00012715657552083334]


def test_init_resolution_res():
    cnfg = zoom.Config(initResolution=2)
    resolutions = [1, 2, 3, 4, 5, 6]
    assert zoom.init_resolution(cnfg, resolutions) == 2


def test_init_resolution_scale():
    cnfg = zoom.Config(initScale=5)
    resolutions = [1, 2, 3, 4, 5, 6]
    assert zoom.init_resolution(cnfg, resolutions) == 1


def test_init_resolution_empty():
    cnfg = zoom.Config()
    resolutions = [1, 2, 3, 4, 5, 6]
    assert zoom.init_resolution(cnfg, resolutions) == 4
