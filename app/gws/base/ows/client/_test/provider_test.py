"""Tests for the base OWS provider."""

import gws
import gws.test.util as u
import gws.base.ows.client.provider as provider
import gws.base.ows.client.request as request

root = u.gws_root()


class _Provider(provider.Object):
    protocol = gws.OwsProtocol.WMS


def _provider(**kwargs):
    return root.create(_Provider, url='http://host_1/ows', **kwargs)


def _caps_op(verb, **kwargs):
    return gws.OwsOperation(verb=verb, url='http://host_1/ows', **kwargs)


def test_configure_operations_from_caps():
    p = _provider()
    p.configure_operations([_caps_op(gws.OwsVerb.GetMap, mimeTypes=['image/png8', 'image/jpeg'])])
    op = p.get_operation(gws.OwsVerb.GetMap)
    assert op.mimeTypes == ['image/png; mode=8bit', 'image/jpeg']
    assert op.mimeTypeMap == {'image/png; mode=8bit': 'image/png8'}
    assert op.defaultMimeType == 'image/png; mode=8bit'


def test_configure_operations_config_overrides_caps():
    p = _provider(operations=[dict(verb=gws.OwsVerb.GetMap, url='http://host_2/ows', formats=['image/jpeg'])])
    p.configure_operations([_caps_op(gws.OwsVerb.GetMap, mimeTypes=['image/png'], params={'a': '1'})])
    op = p.get_operation(gws.OwsVerb.GetMap)
    assert op.url == 'http://host_2/ows'
    assert op.params == {'a': '1'}
    assert op.mimeTypes == ['image/jpeg']
    assert op.defaultMimeType == 'image/jpeg'


def test_configure_operations_config_only():
    p = _provider(operations=[dict(verb=gws.OwsVerb.GetFeatureInfo, url='http://host_2/ows')])
    p.configure_operations([_caps_op(gws.OwsVerb.GetMap)])
    assert sorted(op.verb for op in p.operations) == [gws.OwsVerb.GetFeatureInfo, gws.OwsVerb.GetMap]
    op = p.get_operation(gws.OwsVerb.GetFeatureInfo)
    assert op.url == 'http://host_2/ows'
    assert op.defaultMimeType == gws.lib.mime.GML


def test_get_operation_requires_url():
    p = _provider()
    p.configure_operations([gws.OwsOperation(verb=gws.OwsVerb.GetMap, postUrl='http://host_1/post')])
    assert p.get_operation(gws.OwsVerb.GetMap) is None
    assert p.get_operation(gws.OwsVerb.GetMap, gws.RequestMethod.POST).postUrl == 'http://host_1/post'


def test_prepare_operation_sends_source_spelling():
    p = _provider()
    p.configure_operations([_caps_op(gws.OwsVerb.GetMap, mimeTypes=['image/png;mode=8bit'])])
    op = p.get_operation(gws.OwsVerb.GetMap)
    args = p.prepare_operation(op, params={'format': 'image/png; mode=8bit', 'info_format': 'image/png; mode=8bit', 'x': 'image/png; mode=8bit'})
    assert args.params['FORMAT'] == 'image/png;mode=8bit'
    assert args.params['INFO_FORMAT'] == 'image/png;mode=8bit'
    assert args.params['X'] == 'image/png; mode=8bit'


def test_prepare_operation_unknown_mime_type_unchanged():
    p = _provider()
    p.configure_operations([_caps_op(gws.OwsVerb.GetMap, mimeTypes=['image/png8'])])
    op = p.get_operation(gws.OwsVerb.GetMap)
    args = p.prepare_operation(op, params={'FORMAT': 'image/jpeg'})
    assert args.params['FORMAT'] == 'image/jpeg'


def test_prepare_operation_checks_allowed_values_after_translation():
    p = _provider()
    op = _caps_op(gws.OwsVerb.GetFeature, mimeTypes=['text/xml; subtype=gml/3.1.1'], allowedParameters={'OUTPUTFORMAT': ['text/xml; subtype=gml/3.1.1']})
    p.configure_operations([op])
    op = p.get_operation(gws.OwsVerb.GetFeature)
    args = p.prepare_operation(op, params={'OUTPUTFORMAT': gws.lib.mime.GML3})
    assert args.params['OUTPUTFORMAT'] == 'text/xml; subtype=gml/3.1.1'
    with u.raises(gws.Error):
        p.prepare_operation(op, params={'OUTPUTFORMAT': 'application/json'})


def test_prepare_operation_get_and_post():
    p = _provider()
    p.configure_operations([_caps_op(gws.OwsVerb.GetMap, postUrl='http://host_1/post', params={'map': 'x'})])
    op = p.get_operation(gws.OwsVerb.GetMap)

    args = p.prepare_operation(op, params={'a': '1'})
    assert args.url == 'http://host_1/ows'
    assert args.params == {'map': 'x', 'A': '1'}

    args = p.prepare_operation(op, gws.RequestMethod.POST, params={'a': '1'})
    assert args.url == 'http://host_1/post'
    assert args.params == {'A': '1'}


def _capture_caps_request(monkeypatch):
    calls = []
    monkeypatch.setattr(request, 'get_text', lambda args: calls.append(args) or '<caps/>')
    monkeypatch.setattr(gws.u, 'get_cached_object', lambda name, max_age, fn: fn())
    return calls


def test_get_capabilities_uses_service_url(monkeypatch):
    calls = _capture_caps_request(monkeypatch)
    p = root.create(_Provider, url='http://host_1/ows?map=x')
    assert p.get_capabilities() == '<caps/>'
    assert calls[0].url == 'http://host_1/ows'
    assert calls[0].params == {'map': 'x'}
    assert calls[0].verb == gws.OwsVerb.GetCapabilities


def test_get_capabilities_uses_config(monkeypatch):
    calls = _capture_caps_request(monkeypatch)
    p = root.create(
        _Provider,
        url='http://host_1/ows?map=x',
        operations=[dict(verb=gws.OwsVerb.GetCapabilities, url='http://host_2/caps', params={'b': '2'})],
    )
    p.get_capabilities()
    assert calls[0].url == 'http://host_2/caps'
    assert calls[0].params == {'b': '2'}
