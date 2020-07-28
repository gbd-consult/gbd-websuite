import _test.util as u


def test_get_capabilities():
    r = u.req('_/cmd/owsHttpService/uid/csw', params={
        'SERVICE': 'CSW',
        'REQUEST': 'GetCapabilities'
    })

    a, b = u.compare_xml(r, path='/data/response_xml/csw_GetCapabilities.xml')
    assert a == b
