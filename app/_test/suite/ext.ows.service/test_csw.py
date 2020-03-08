import _test.util as u


def test_get_capabilities():
    r = u.req('_/cmd/owsHttpGetCsw', params={
        'SERVICE': 'CSW',
        'REQUEST': 'GetCapabilities'
    })

    assert u.xml(r) == u.xml('/data/response_xml/csw_GetCapabilities.xml')