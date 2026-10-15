"""Tests for the QGIS provider requests, without QGIS Server."""

import gws
import gws.test.util as u
import gws.lib.crs
import gws.lib.shape
import gws.plugin.qgis.provider as provider

CRS_3857 = gws.lib.crs.get(3857)


def _provider(calls):
    p = provider.Object()
    p.alwaysXY = False
    p.forceCrs = CRS_3857
    p.call_server = lambda params: calls.append(params) or gws.Data(content_type='image/png', content=b'png_1', text='')
    return p


def _get_map(p, mime_type=''):
    bounds = gws.Bounds(crs=CRS_3857, extent=(0, 0, 100, 100))
    return p.get_map(bounds, 10, 10, {'LAYERS': ['layer_1']}, mime_type)


def _get_features(p, mime_type=''):
    search = gws.SearchQuery(shape=gws.lib.shape.from_xy(50, 50, CRS_3857), resolution=1)
    return p.get_features(search, [gws.SourceLayer(name='layer_1')], mime_type)


def test_get_map_default_mime_type():
    calls = []
    assert _get_map(_provider(calls)) == b'png_1'
    assert calls[0]['FORMAT'] == gws.lib.mime.PNG


def test_get_map_explicit_mime_type():
    calls = []
    _get_map(_provider(calls), gws.lib.mime.PNG8)
    assert calls[0]['FORMAT'] == 'image/png; mode=8bit'


def test_get_map_params_override_mime_type():
    calls = []
    p = _provider(calls)
    p.get_map(gws.Bounds(crs=CRS_3857, extent=(0, 0, 100, 100)), 10, 10, {'FORMAT': 'value_1'}, gws.lib.mime.PNG8)
    assert calls[0]['FORMAT'] == 'value_1'


def test_get_features_default_mime_type():
    calls = []
    assert _get_features(_provider(calls)) == []
    assert calls[0]['INFO_FORMAT'] == gws.lib.mime.XML


def test_get_features_explicit_mime_type():
    calls = []
    _get_features(_provider(calls), gws.lib.mime.HTML)
    assert calls[0]['INFO_FORMAT'] == 'text/html'
