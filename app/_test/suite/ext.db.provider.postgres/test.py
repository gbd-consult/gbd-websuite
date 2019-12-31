import gws.gis.shape
import gws.tools.json2

import _test.util as u
import _test.common.const as cc


def test_find_points():
    x = cc.POI.tour_eiffel_3857_x
    y = cc.POI.tour_eiffel_3857_y

    bbox = (x, y, x + 101, y + 101,)

    sh = gws.gis.shape.from_extent(bbox, 'EPSG:3857')

    r = u.cmd('searchFindFeatures', {
        'projectUid': 'a',
        'layerUids': ['a.map.points_3857'],
        'resolution': 1,
        'shapes': [sh.props]
    })

    r = r.json()

    assert len(r['features']) == 4

    r = r['features']

    assert r[0]['uid'] == 'a.map.points_3857___1'
    assert r[1]['uid'] == 'a.map.points_3857___2'
    assert r[2]['uid'] == 'a.map.points_3857___6'
    assert r[3]['uid'] == 'a.map.points_3857___7'

    assert r[0]['attributes'] == [{'name': 'p_str', 'value': 'p_str_1', }, {'name': 'p_int', 'value': 100, }, {'name': 'p_date', 'value': '2019-01-01', }]
    assert r[1]['attributes'] == [{'name': 'p_str', 'value': 'p_str_2', }, {'name': 'p_int', 'value': 200, }, {'name': 'p_date', 'value': '2019-01-02', }]
    assert r[2]['attributes'] == [{'name': 'p_str', 'value': 'p_str_6', }, {'name': 'p_int', 'value': 600, }, {'name': 'p_date', 'value': '2019-01-06', }]
    assert r[3]['attributes'] == [{'name': 'p_str', 'value': 'p_str_7', }, {'name': 'p_int', 'value': 700, }, {'name': 'p_date', 'value': '2019-01-07', }]

    assert r[0]['shape']['geometry']['coordinates'] == [x, y]
    assert r[1]['shape']['geometry']['coordinates'] == [x + 100, y]
    assert r[2]['shape']['geometry']['coordinates'] == [x, y + 100]
    assert r[3]['shape']['geometry']['coordinates'] == [x + 100, y + 100]


def test_render_squares():
    bbox = [
        cc.POI.flatiron_building_3857_x,
        cc.POI.flatiron_building_3857_y,
        cc.POI.flatiron_building_3857_x + 350,
        cc.POI.flatiron_building_3857_y + 350,
    ]

    url = '_/cmd/mapHttpGetBbox/layerUid/a.map.squares_3857/bbox/' + gws.as_str_list(bbox)

    r = u.req(url, params={'width': 200, 'height': 200})
    d = u.compare_image_response(r, '/data/squares_200x200.png')
    assert not d

    r = u.req(url, params={'width': 400, 'height': 400})
    d = u.compare_image_response(r, '/data/squares_400x400.png')
    assert not d

    r = u.req(url, params={'width': 800, 'height': 400})
    d = u.compare_image_response(r, '/data/squares_800x400.png')
    assert not d


def test_render_squares_styled():
    bbox = [
        cc.POI.flatiron_building_3857_x,
        cc.POI.flatiron_building_3857_y,
        cc.POI.flatiron_building_3857_x + 350,
        cc.POI.flatiron_building_3857_y + 350,
    ]

    url = '_/cmd/mapHttpGetBbox/layerUid/a.map.squares_3857_styled/bbox/' + gws.as_str_list(bbox)

    r = u.req(url, params={'width': 200, 'height': 200})
    d = u.compare_image_response(r, '/data/squares_styled_200x200.png')
    assert not d


def test_render_squares_reprojected():
    bbox = [
        cc.POI.big_ben_3857_x,
        cc.POI.big_ben_3857_y,
        cc.POI.big_ben_3857_x + 350,
        cc.POI.big_ben_3857_y + 350,
    ]

    url = '_/cmd/mapHttpGetBbox/layerUid/a.map.squares_25832/bbox/' + gws.as_str_list(bbox)

    r = u.req(url, params={'width': 200, 'height': 200})
    d = u.compare_image_response(r, '/data/squares_reprojected_200x200.png')
    assert not d
