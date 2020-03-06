import gws.gis.shape
import gws.gis.proj
import gws.tools.json2
import gws.gis.extent

import _test.util as u
import _test.common.const as cc


def test_search():
    x, y = cc.POINTS.ghana

    # should give us point 14
    x += 300
    y += 200

    x, y = gws.gis.proj.transform_xy(x, y, cc.CRS_3857, cc.CRS_25832)
    shape = gws.gis.shape.from_xy(x, y, cc.CRS_25832)

    r = u.cmd('searchFindFeatures', {
        'projectUid': 'a',
        'layerUids': ['a.map.qgis_flat_ghana'],
        'shapes': [shape.props],
        'crs': 'EPSG:25832',
        'resolution': 1,
    })

    r = r.json()

    exp = [
        {
            "attributes": "id=<14> p_date=<2019-01-14T00:00:00> p_int=<1400> p_str=<points_ghana_25832/14>",
            "geometry": "POINT EPSG:25832",
            "uid": "a.map.qgis_flat_ghana___14"
        }
    ]

    assert u.short_features(r['features']) == exp


def test_render():
    x, y = cc.POINTS.ghana

    x, y = gws.gis.proj.transform_xy(x, y, cc.CRS_3857, cc.CRS_25833)
    bbox = x, y, x + 350, y + 350
    bbox = gws.gis.extent.transform(bbox, cc.CRS_25833, cc.CRS_3857)

    url = '_/cmd/mapHttpGetBox/layerUid/a.map.qgis_flat_ghana/bbox/' + gws.as_str_list(bbox)

    r = u.req(url, params={'width': 300, 'height': 300})
    assert True is u.response_image_matches(r, '/data/response_images/ghana_200x200.png')


def test_search_multi():
    x, y = cc.POINTS.dus2

    x += 305
    y += 405

    shape = gws.gis.shape.from_xy(x, y, cc.CRS_3857)

    r = u.cmd('searchFindFeatures', {
        'projectUid': 'a',
        'layerUids': ['a.map.qgis_flat_dus'],
        'shapes': [shape.props],
        'resolution': 1,
    })

    r = r.json()

    exp = [
        {
            "attributes": "id=<23> p_str=<squares_dus1_3857/23> p_int=<2300> p_date=<2019-01-23>",
            "geometry": "POLYGON EPSG:3857",
            "uid": "a.map.qgis_flat_dus___23"
        },
        {
            "attributes": "id=<24> p_str=<squares_dus2_3857/24> p_int=<2400> p_date=<2019-01-24>",
            "geometry": "POLYGON EPSG:3857",
            "uid": "a.map.qgis_flat_dus___24"
        }
    ]

    assert u.short_features(r['features']) == exp


def test_render_multi():
    x, y = cc.POINTS.dus2
    bbox = x, y, x + 350, y + 450

    url = '_/cmd/mapHttpGetBox/layerUid/a.map.qgis_flat_dus/bbox/' + gws.as_str_list(bbox)

    r = u.req(url, params={'width': 300, 'height': 300})
    assert True is u.response_image_matches(r, '/data/response_images/dus_200x200.png')
