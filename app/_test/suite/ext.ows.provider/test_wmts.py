import gws.gis.shape
import gws.gis.proj
import gws.tools.json2
import gws.gis.extent

import _test.util as u
import _test.common.const as cc


def test_render_box():
    x, y = cc.POINTS.mexico
    bbox = (x - 300, y - 150, x, y + 150)

    url = '_/cmd/mapHttpGetBox/layerUid/a.map.wmts_squares_mexico_25832/bbox/' + u.strlist(bbox)

    r = u.req(url, params={'width': 100, 'height': 100})
    a, b = u.compare_image(r, '/data/response_images/wmts_points_100x100.png')
    assert a == b

    r = u.req(url, params={'width': 400, 'height': 400})
    a, b = u.compare_image(r, '/data/response_images/wmts_points_400x400.png')
    assert a == b


def test_render_tile():
    url = '_/cmd/mapHttpGetXyz/layerUid/a.map.wmts_squares_mexico_25832/z/12/x/%d/y/%d/t.png'

    r = u.req(url % (0, 0))
    a, b = u.compare_image(r, '/data/response_images/wmts_tile_12_0_0.png')
    assert a == b

    r = u.req(url % (1, 0))
    a, b = u.compare_image(r, '/data/response_images/wmts_tile_12_1_0.png')
    assert a == b

    r = u.req(url % (0, 1))
    a, b = u.compare_image(r, '/data/response_images/wmts_tile_12_0_1.png')
    assert a == b

    r = u.req(url % (1, 1))
    a, b = u.compare_image(r, '/data/response_images/wmts_tile_12_1_1.png')
    assert a == b
