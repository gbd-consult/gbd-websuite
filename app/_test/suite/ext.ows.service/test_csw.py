import _test.util as u


def test_get_capabilities():
    r = u.req('_/cmd/owsHttp', params={
        'serviceName': 'csw',
        'SERVICE': 'CSW',
        'REQUEST': 'GetCapabilities'
    })

    assert True is u.response_xml_matches(r, path='/data/response_xml/csw_GetCapabilities.xml')