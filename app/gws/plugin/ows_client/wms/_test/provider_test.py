"""Tests for the WMS provider requests, without a service."""

import gws
import gws.test.util as u
import gws.base.ows.client.request as request
import gws.lib.crs
import gws.lib.shape
import gws.plugin.ows_client.wms.provider as provider

CRS_3857 = gws.lib.crs.get(3857)


def _provider(get_map_mime_types=None, get_feature_info_mime_types=None):
    p = provider.Object()
    p.config = gws.Config()
    p.url = 'http://host_1/wms'
    p.version = '1.3.0'
    p.alwaysXY = False
    p.forceCrs = CRS_3857
    p.authorization = None
    p.configure_operations([
        gws.OwsOperation(verb=gws.OwsVerb.GetMap, url=p.url, mimeTypes=get_map_mime_types or []),
        gws.OwsOperation(verb=gws.OwsVerb.GetFeatureInfo, url=p.url, mimeTypes=get_feature_info_mime_types or []),
    ])
    return p


@u.fixture
def calls(monkeypatch):
    calls = []
    monkeypatch.setattr(request, 'get', lambda args: calls.append(args) or gws.Data(content_type='image/png', content=b'png_1'))
    monkeypatch.setattr(request, 'get_text', lambda args: calls.append(args) or '')
    return calls


def _get_map(p, mime_type=''):
    bounds = gws.Bounds(crs=CRS_3857, extent=(0, 0, 100, 100))
    return p.get_map(bounds, 10, 10, [gws.SourceLayer(name='layer_1')], mime_type)


def _get_features(p, mime_type=''):
    search = gws.SearchQuery(shape=gws.lib.shape.from_xy(50, 50, CRS_3857), resolution=1)
    return p.get_features(search, [gws.SourceLayer(name='layer_1')], mime_type)


def test_get_map_default_mime_type(calls):
    p = _provider(get_map_mime_types=['image/jpeg', 'image/png8'])
    assert _get_map(p) == b'png_1'
    assert calls[0].params['FORMAT'] == 'image/png8'


def test_get_map_explicit_mime_type(calls):
    p = _provider(get_map_mime_types=['image/jpeg', 'image/png8'])
    _get_map(p, gws.lib.mime.JPEG)
    assert calls[0].params['FORMAT'] == 'image/jpeg'


def test_get_map_explicit_mime_type_in_source_spelling(calls):
    p = _provider(get_map_mime_types=['image/png;mode=8bit'])
    _get_map(p, gws.lib.mime.PNG8)
    assert calls[0].params['FORMAT'] == 'image/png;mode=8bit'


def test_get_map_rejects_non_image(monkeypatch):
    monkeypatch.setattr(request, 'get', lambda args: gws.Data(content_type='text/xml', content=b'<x/>'))
    p = _provider(get_map_mime_types=['image/png'])
    with u.raises(gws.ExternalServiceError):
        _get_map(p)


def test_get_features_default_mime_type(calls):
    p = _provider(get_feature_info_mime_types=['text/html', 'application/vnd.ogc.gml'])
    assert _get_features(p) == []
    assert calls[0].params['INFO_FORMAT'] == 'application/vnd.ogc.gml'


def test_get_features_explicit_mime_type(calls):
    p = _provider(get_feature_info_mime_types=['text/html', 'application/vnd.ogc.gml'])
    _get_features(p, gws.lib.mime.HTML)
    assert calls[0].params['INFO_FORMAT'] == 'text/html'
