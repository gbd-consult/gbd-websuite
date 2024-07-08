"""Tests for the crs module."""

import gws
import gws.test.util as u
import gws.gis.crs as crs


def test_get_no_name():
    assert not crs.get('')


def test_get():
    assert str(crs.get('EPSG:3857')) == '<crs:3857>'


def test_parse():
    assert crs.parse('EPSG:3857') == ('epsg', crs.get('3857'))


def test_parse_srid():
    assert crs.parse('3857') == ('srid', crs.get('3857'))


def test_parse_no_format():
    assert crs.parse('11wrongFormat') == ('', None)


def test_require():
    assert crs.require('3857') == crs.get('3857')


def test_require_exception():
    with u.raises(Exception):
        crs.require('FOOBAR')


def test_best_match():
    lst = [crs.WEBMERCATOR, crs.WGS84]
    assert crs.best_match(crs.WGS84, lst) == crs.get('4326')


def test_best_match_not__list():
    lst = [crs.WEBMERCATOR]
    assert crs.best_match(crs.WGS84, lst) == crs.get('3857')


def test_best_bounds():
    b1 = gws.Bounds(crs=crs.WEBMERCATOR, extent=(0.0, 1.0, 1.0, 0.0))
    b2 = gws.Bounds(crs=crs.WGS84, extent=(0.0, 1.0, 1.0, 0.0))
    lst = [b1, b2]
    assert crs.best_bounds(crs.WGS84, lst) == b2


def test_best_axis():
    assert crs.best_axis(crs.WGS84) == 'xy'


def test_best_axis_inverted():
    assert crs.best_axis(crs.WGS84, inverted_crs=[crs.WGS84]) == 'yx'
