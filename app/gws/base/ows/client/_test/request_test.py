# import gws.base.ows.client.request
# import gws.gis.ows.error
# import gws.lib.test as test
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
