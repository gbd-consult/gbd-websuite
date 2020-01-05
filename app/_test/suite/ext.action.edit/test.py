import _test.util as u
import _test.common.const as cc


def test_insert():
    x, y = cc.POINTS.paris

    r = u.cmd('editAddFeatures', {
        'layerUid': 'a.map.paris_3857',
        'features': [
            {
                'attributes': [
                    {'name': 'p_str1', 'value': 'p_str1_new_1'},
                    {'name': 'p_str2', 'value': 'p_str2_new_1'},
                    {'name': 'p_int1', 'value': 111},
                    {'name': 'p_str2', 'value': 222},
                    {'name': 'p_date', 'value': '2020-03-01'},
                ],
                'shape': {'crs': 'EPSG:3857', 'geometry': {'type': 'Point', 'coordinates': [x + 1000 * 1, y + 1000 * 2]}}
            },
            {
                'attributes': [
                    {'name': 'p_str1', 'value': 'p_str1_new_2'},
                    {'name': 'p_str2', 'value': 'p_str2_new_2'},
                    {'name': 'p_int1', 'value': 1111},
                    {'name': 'p_str2', 'value': 2222},
                    {'name': 'p_date', 'value': '2020-03-02'},
                ],
                'shape': {'crs': 'EPSG:3857', 'geometry': {'type': 'Point', 'coordinates': [x + 1000 * 2, y + 1000 * 2]}}
            },
        ]
    })

    r = r.json()

    exp = [
        {
            "attributes": "id=<51> p_str1=<p_str1_new_1> p_str2=<222> p_int1=<111> p_int2=<None> p_date=<2020-03-01>",
            "geometry": "POINT EPSG:3857",
            "uid": "a.map.paris_3857___51"
        },
        {
            "attributes": "id=<52> p_str1=<p_str1_new_2> p_str2=<2222> p_int1=<1111> p_int2=<None> p_date=<2020-03-02>",
            "geometry": "POINT EPSG:3857",
            "uid": "a.map.paris_3857___52"
        }
    ]

    assert u.short_features(r['features']) == exp

    exp = [
        {
            "id": 50,
            "p_str1": "paris_3857/50"
        },
        {
            "id": 51,
            "p_str1": "p_str1_new_1"
        },
        {
            "id": 52,
            "p_str1": "p_str1_new_2"
        }
    ]

    recs = u.postgres_select(f'SELECT id, p_str1 FROM paris_3857 WHERE id >= 50 ORDER BY id')
    assert recs == exp


def test_update():
    x, y = cc.POINTS.paris

    r = u.cmd('editUpdateFeatures', {
        'layerUid': 'a.map.paris_3857',
        'features': [
            {
                'uid': '5',
                'attributes': [
                    {'name': 'p_str1', 'value': 'p_str1_UPDATE_5'},
                ],
            },
            {
                'uid': 'a.map.paris_3857___15',
                'attributes': [
                    {'name': 'p_str1', 'value': 'p_str1_UPDATE_15'},
                ],
                'shape': {'crs': 'EPSG:3857', 'geometry': {'type': 'Point', 'coordinates': [x + 1000 * 15, y + 1000 * 15]}}
            },
        ]
    })

    r = r.json()

    exp = [
        {
            "attributes": "id=<5> p_str1=<p_str1_UPDATE_5> p_str2=<paris_3857/5> p_int1=<500> p_int2=<500> p_date=<2019-01-05>",
            "geometry": "POINT EPSG:3857",
            "uid": "a.map.paris_3857___5"
        },
        {
            "attributes": "id=<15> p_str1=<p_str1_UPDATE_15> p_str2=<paris_3857/15> p_int1=<1500> p_int2=<1500> p_date=<2019-01-15>",
            "geometry": "POINT EPSG:3857",
            "uid": "a.map.paris_3857___15"
        }
    ]

    assert u.short_features(r['features']) == exp

    exp = [
        {
            "id": 5,
            "p_str1": "p_str1_UPDATE_5"
        },
        {
            "id": 15,
            "p_str1": "p_str1_UPDATE_15"
        },
    ]

    recs = u.postgres_select(f'SELECT id, p_str1 FROM paris_3857 WHERE id IN (5, 15) ORDER BY id')
    assert recs == exp


def test_delete():
    x, y = cc.POINTS.paris

    r = u.cmd('editDeleteFeatures', {
        'layerUid': 'a.map.paris_3857',
        'features': [
            {
                'uid': '6',
            },
            {
                'uid': 'a.map.paris_3857___16',
            },
        ]
    })

    r = r.json()

    assert u.short_features(r['features']) == []

    recs = u.postgres_select(f'SELECT id, p_str1 FROM paris_3857 WHERE id IN (6, 16) ORDER BY id')
    assert recs == []
