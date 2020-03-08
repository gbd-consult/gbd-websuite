import gws.gis.shape
import gws.gis.proj
import gws.tools.json2
import gws.gis.extent

import _test.util as u
import _test.common.const as cc


def test_search_points_in_provider_projection():
    x, y = cc.POINTS.ghana

    # should give us point 14
    x += 300
    y += 200

    x, y = gws.gis.proj.transform_xy(x, y, cc.CRS_3857, cc.CRS_25832)
    shape = gws.gis.shape.from_xy(x, y, cc.CRS_25832)

    r = u.cmd('searchFindFeatures', {
        'projectUid': 'a',
        'layerUids': ['a.map.wms_points_ghana_25832'],
        'crs': cc.CRS_25832,
        'shapes': [shape.props],
        'resolution': 1,
    })

    r = r.json()

    exp = [
        {
            "attributes": "gml_id=<points_ghana_25832.14> id=<14> p_date=<2019-01-14T00:00:00> p_int=<1400> p_str=<points_ghana_25832/14>",
            "uid": "a.map.wms_points_ghana_25832___14"
        }
    ]

    assert u.short_features(r['features']) == exp


def test_search_points_with_reprojection():
    x, y = cc.POINTS.ghana

    # should give us point 14
    x += 300
    y += 200

    shape = gws.gis.shape.from_xy(x, y, cc.CRS_3857)

    r = u.cmd('searchFindFeatures', {
        'projectUid': 'a',
        'layerUids': ['a.map.wms_points_ghana_25832'],
        'crs': cc.CRS_3857,
        'shapes': [shape.props],
        'resolution': 1,
    })

    r = r.json()

    exp = [
        {
            "attributes": "gml_id=<points_ghana_25832.14> id=<14> p_date=<2019-01-14T00:00:00> p_int=<1400> p_str=<points_ghana_25832/14>",
            "uid": "a.map.wms_points_ghana_25832___14"
        }
    ]

    assert u.short_features(r['features']) == exp


def test_render():
    x, y = cc.POINTS.ghana
    bbox = (x + 50, y + 50, x + 350, y + 350,)

    url = '_/cmd/mapHttpGetBox/layerUid/a.map.wms_points_ghana_25832/bbox/' + u.strlist(bbox)

    r = u.req(url, params={'width': 100, 'height': 100})
    assert True is u.response_image_matches(r, '/data/response_images/wms_points_100x100.png')

    r = u.req(url, params={'width': 400, 'height': 400})
    assert True is u.response_image_matches(r, '/data/response_images/wms_points_400x400.png')
