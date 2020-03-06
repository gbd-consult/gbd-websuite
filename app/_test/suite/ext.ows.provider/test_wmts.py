import gws.gis.shape
import gws.gis.proj
import gws.tools.json2
import gws.gis.extent

import _test.util as u
import _test.common.const as cc


def test_render_box():
    x, y = cc.POINTS.mexico
    bbox = (x - 300, y - 150, x, y + 150)

    url = '_/cmd/mapHttpGetBox/layerUid/a.map.wmts_squares_mexico_25832/bbox/' + gws.as_str_list(bbox)

    r = u.req(url, params={'width': 100, 'height': 100})
    assert True is u.response_image_matches(r, '/data/response_images/wmts_points_100x100.png')

    r = u.req(url, params={'width': 400, 'height': 400})
    assert True is u.response_image_matches(r, '/data/response_images/wmts_points_400x400.png')


def test_render_tile():
    url = '_/cmd/mapHttpGetXyz/layerUid/a.map.wmts_squares_mexico_25832/z/12/x/%d/y/%d/t.png'

    r = u.req(url % (0, 0))
    assert True is u.response_image_matches(r, '/data/response_images/wmts_tile_12_0_0.png')

    r = u.req(url % (1, 0))
    assert True is u.response_image_matches(r, '/data/response_images/wmts_tile_12_1_0.png')

    r = u.req(url % (0, 1))
    assert True is u.response_image_matches(r, '/data/response_images/wmts_tile_12_0_1.png')

    r = u.req(url % (1, 1))
    assert True is u.response_image_matches(r, '/data/response_images/wmts_tile_12_1_1.png')
