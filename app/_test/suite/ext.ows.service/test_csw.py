import _test.util as u


def test_get_capabilities():
    r = u.req('_/cmd/owsHttp', params={
        'serviceName': 'csw',
        'SERVICE': 'CSW',
        'REQUEST': 'GetCapabilities'
    })

    assert u.xml(r.text) == u.read('/data/csw_GetCapabilities.xml')
