"""Tests for the WFS provider requests, without a service."""

import gws
import gws.test.util as u
import gws.base.ows.client.request as request
import gws.lib.crs
import gws.plugin.ows_client.wfs.provider as provider

CRS_3857 = gws.lib.crs.get(3857)


def _provider(mime_types):
    p = provider.Object()
    p.config = gws.Config()
    p.url = 'http://host_1/wfs'
    p.version = '2.0.0'
    p.isWfs2 = True
    p.withBboxCrs = False
    p.alwaysXY = False
    p.forceCrs = CRS_3857
    p.authorization = None
    p.configure_operations([gws.OwsOperation(verb=gws.OwsVerb.GetFeature, url=p.url, mimeTypes=mime_types)])
    return p


@u.fixture
def calls(monkeypatch):
    calls = []
    monkeypatch.setattr(request, 'get_text', lambda args: calls.append(args) or '')
    return calls


def _get_features(p, mime_type=''):
    search = gws.SearchQuery(bounds=gws.Bounds(crs=CRS_3857, extent=(0, 0, 100, 100)))
    return p.get_features(search, [gws.SourceLayer(name='layer_1')], mime_type)


def test_get_features_default_mime_type(calls):
    p = _provider(['application/json', 'application/gml+xml; version=3.2'])
    assert _get_features(p) == []
    assert calls[0].params['OUTPUTFORMAT'] == 'application/gml+xml; version=3.2'


def test_get_features_default_in_source_spelling(calls):
    p = _provider(['application/json', 'text/xml; subtype=gml/3.1.1'])
    _get_features(p)
    assert calls[0].params['OUTPUTFORMAT'] == 'text/xml; subtype=gml/3.1.1'


def test_get_features_explicit_mime_type(calls):
    p = _provider(['application/json', 'text/xml; subtype=gml/3.1.1'])
    _get_features(p, gws.lib.mime.JSON)
    assert calls[0].params['OUTPUTFORMAT'] == 'application/json'


def test_get_features_extra_params_win(calls):
    p = _provider(['application/json', 'text/xml; subtype=gml/3.1.1'])
    search = gws.SearchQuery(bounds=gws.Bounds(crs=CRS_3857, extent=(0, 0, 100, 100)), extraParams={'outputFormat': 'value_1'})
    p.get_features(search, [gws.SourceLayer(name='layer_1')])
    assert calls[0].params['OUTPUTFORMAT'] == 'value_1'
