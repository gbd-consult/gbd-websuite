import gws.gis.shape
import gws.tools.json2

import _test.util as u


def test_find_points():
    # start_x=1000, start_y=2000, gap=100,

    bbox = [1000, 2000, 1101, 2101, ]

    sh = gws.gis.shape.from_bbox(bbox, 'EPSG:3857')

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

    assert r[0]['shape']['geometry']['coordinates'] == [1000.0, 2000.0]
    assert r[1]['shape']['geometry']['coordinates'] == [1100.0, 2000.0]
    assert r[2]['shape']['geometry']['coordinates'] == [1000.0, 2100.0]
    assert r[3]['shape']['geometry']['coordinates'] == [1100.0, 2100.0]
