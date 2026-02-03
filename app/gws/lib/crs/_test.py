"""Tests for the crs module."""

import gws
import gws.test.util as u
import gws.lib.crs as crs


def test_get_no_name():
    assert not crs.get('')


def test_get_unknown():
    assert crs.get('EPSG:0') is None
    assert crs.get('FOOBAR') is None


def test_get():
    assert str(crs.get('EPSG:3857')) == '<crs:3857>'
    assert crs.get('EPSG:3857') == crs.WEBMERCATOR


def test_parse():
    fmt, obj = crs.parse('EPSG:3857')
    assert fmt == gws.CrsFormat.epsg
    assert obj == crs.get('3857')


def test_parse_srid():
    fmt, obj = crs.parse('3857')
    assert fmt == gws.CrsFormat.srid
    assert obj == crs.get('3857')


def test_parse_aliases():
    fmt, obj = crs.parse('CRS:84')
    assert fmt == gws.CrsFormat.crs
    assert obj == crs.WGS84

    fmt, obj = crs.parse('epsg:900913')
    assert fmt == gws.CrsFormat.epsg
    assert obj == crs.WEBMERCATOR


def test_parse_no_format():
    fmt, obj = crs.parse('11wrongFormat')
    assert fmt == gws.CrsFormat.none
    assert obj is None


def test_require():
    assert crs.require('3857') == crs.get('3857')


def test_require_exception():
    with u.raises(crs.Error):
        crs.require('FOOBAR')


def test_best_match():
    lst = [crs.WEBMERCATOR, crs.WGS84]
    assert crs.best_match(crs.WGS84, lst) == crs.get('4326')


def test_best_match_not__list():
    lst = [crs.WEBMERCATOR]
    assert crs.best_match(crs.WGS84, lst) == crs.get('3857')


def test_axis_for_format():
    # WGS84 is internally YX, but output axis depends on the requested CRS string format.
    assert crs.WGS84.axis_for_format(gws.CrsFormat.epsg) == gws.Axis.xy
    assert crs.WGS84.axis_for_format(gws.CrsFormat.urn) == gws.Axis.yx


def test_transform_extent():
    got = crs.WGS84.transform_extent((0.0, 1.0, 1.0, 0.0), crs_to=crs.WEBMERCATOR)
    exp = (0.0, 0.0, 111319.49079327357, 111325.1428663851)
    assert u.is_close(got, exp, abs_tol=1e-4)


def test_transformer():
    tr = crs.WGS84.transformer(crs.WEBMERCATOR)
    assert callable(tr)
    x, y = tr(1.0, 1.0)  # lon/lat -> meters (always_xy=True)
    assert u.is_close(x, 111319.49079327357, abs_tol=1e-4)
    assert u.is_close(y, 111325.1428663851, abs_tol=1e-4)


def test_to_string():
    assert crs.WEBMERCATOR.to_string() == 'EPSG:3857'
    assert crs.WEBMERCATOR.to_string(gws.CrsFormat.epsg) == 'EPSG:3857'
    assert crs.WEBMERCATOR.to_string(gws.CrsFormat.srid) == '3857'


def test_to_geojson():
    assert crs.WEBMERCATOR.to_geojson() == {
        'properties': {'name': 'urn:ogc:def:crs:EPSG::3857'},
        'type': 'name',
    }


def test_extent_size_in_meters_projected():
    # 1000m x 2000m rectangle in WEBMERCATOR
    ext = (0, 0, 1000, 2000)
    w, h = crs.WEBMERCATOR.extent_size_in_meters(ext)
    assert u.is_close(w, 1000, abs_tol=1e-6)
    assert u.is_close(h, 2000, abs_tol=1e-6)


def test_extent_size_in_meters_geographic():
    # 1 degree longitude at equator ≈ 111319.5m, 1 degree latitude ≈ 110574m
    ext = (0, 0, 1, 1)
    w, h = crs.WGS84.extent_size_in_meters(ext)
    assert u.is_close(w, 111319.5, rel_tol=1e-3)
    assert u.is_close(h, 110574, rel_tol=1e-3)
