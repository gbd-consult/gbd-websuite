# import gws.base.ows.client.request
# import gws.gis.ows.error
# import gws.test.util as u
#
#
# def test_request_ok():
#     test.web_server_poke('ok', {'text': 'hello'})
#     test.web_server_begin_capture()
#     gws.base.ows.client.request.get(test.web_server_url('ok'), gws.OwsProtocol.WMS, gws.OwsVerb.GetCapabilities)
#     urls = test.web_server_end_capture()
#     assert urls[0].query == 'SERVICE=WMS&REQUEST=GetCapabilities'
#
#
# def test_request_failed_with_http_status():
#     with test.raises(gws.gis.ows.error.Error):
#         gws.base.ows.client.request.get(test.web_server_url('NOT_FOUND'), gws.OwsProtocol.WMS, gws.OwsVerb.GetCapabilities)
#
#
# def test_request_failed_with_content_error_message():
#     test.web_server_poke('bad', {'text': '<ServiceException>ERROR</ServiceException>'})
#     with test.raises(gws.gis.ows.error.Error):
#         gws.base.ows.client.request.get(test.web_server_url('bad'), gws.OwsProtocol.WMS, gws.OwsVerb.GetCapabilities)


import gws
import gws.base.ows.client.request as request


def test_get_keeps_param_names(monkeypatch):
    calls = []
    monkeypatch.setattr(request, 'get_url', lambda url, **kwargs: calls.append(kwargs) or b'')
    args = request.Args(
        url='http://host_1/ows',
        protocol=gws.OwsProtocol.WMS,
        verb=gws.OwsVerb.GetMap,
        version='1.3.0',
        params={'map': 'map_1', 'LAYERS': 'layer_1'},
    )
    request.get(args)
    assert calls[0]['params'] == {
        'SERVICE': 'WMS',
        'REQUEST': gws.OwsVerb.GetMap,
        'VERSION': '1.3.0',
        'map': 'map_1',
        'LAYERS': 'layer_1',
    }
