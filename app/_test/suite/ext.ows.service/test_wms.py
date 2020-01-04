import gws.gis.shape
import gws.gis.proj
import gws.tools.json2
import gws.gis.extent

import _test.util as u
import _test.common.const as cc



def test_get_capabilities():
    r = u.req('_/cmd/owsHttp', params={
        'projectUid': 'wms1',
        'serviceName': 'wms',
        'SERVICE': 'WMS',
        'REQUEST': 'GetCapabilities'
    })

    assert u.xml(r.text) == u.read('/data/wms_get_capabilities_wms1.xml')

def test_get_map():
    x, y = cc.POINTS.paris

    r = u.req('_/cmd/owsHttp', params={
        'projectUid': 'wms1',
        'serviceName': 'wms',
        'SERVICE': 'WMS',
        'REQUEST': 'GetMap',
        'LAYERS': 'paris_3857',
        'BBOX': u.strlist([x, y, x + 350, y + 350]),
        'WIDTH': 400,
        'HEIGHT': 400,
    })

    d = u.compare_image_response(r, '/data/wms_paris_3857_400x400.png')
    assert not d
