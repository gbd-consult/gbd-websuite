import gws.gis.shape
import gws.gis.proj
import gws.tools.json2
import gws.gis.extent

import _test.util as u
import _test.common.const as cc

_REPROJECTION_ERROR_TOLERANCE = 10


def test_get_points_in_provider_projection():
    x, y = cc.POINTS.ghana
    x, y = gws.gis.proj.transform_xy(x, y, cc.CRS_3857, cc.CRS_25832)
    bbox = (x, y, x + 101, y + 101,)

    r = u.cmd('mapGetFeatures', {
        'projectUid': 'a',
        'layerUid': 'a.map.wfs_points_ghana_25832',
        'crs': cc.CRS_25832,
        'bbox': bbox
    })

    r = r.json()

    exp = [
        {
            "attributes": "gml_id=<points_ghana_25832.1> id=<1> p_date=<2019-01-01 00:00:00> p_int=<100> p_str=<p_str_1>",
            "geometry": f"POINT EPSG:25832",
            "uid": "a.map.wfs_points_ghana_25832___1"
        },
        {
            "attributes": "gml_id=<points_ghana_25832.2> id=<2> p_date=<2019-01-02 00:00:00> p_int=<200> p_str=<p_str_2>",
            "geometry": f"POINT EPSG:25832",
            "uid": "a.map.wfs_points_ghana_25832___2"
        },
        {
            "attributes": "gml_id=<points_ghana_25832.6> id=<6> p_date=<2019-01-06 00:00:00> p_int=<600> p_str=<p_str_6>",
            "geometry": f"POINT EPSG:25832",
            "uid": "a.map.wfs_points_ghana_25832___6"
        },
        {
            "attributes": "gml_id=<points_ghana_25832.7> id=<7> p_date=<2019-01-07 00:00:00> p_int=<700> p_str=<p_str_7>",
            "geometry": f"POINT EPSG:25832",
            "uid": "a.map.wfs_points_ghana_25832___7"
        }
    ]

    assert u.short_features(r['features']) == exp


def test_find_points_in_provider_projection():
    x, y = cc.POINTS.ghana
    x, y = gws.gis.proj.transform_xy(x, y, cc.CRS_3857, cc.CRS_25832)
    bbox = (x, y, x + 100, y + 100,)
    bbox = gws.gis.extent.buffer(bbox, _REPROJECTION_ERROR_TOLERANCE)
    shape = gws.gis.shape.from_extent(bbox, cc.CRS_25832)

    r = u.cmd('searchFindFeatures', {
        'projectUid': 'a',
        'layerUids': ['a.map.wfs_points_ghana_25832'],
        'crs': cc.CRS_25832,
        'resolution': 1,
        'shapes': [shape.props]
    })

    r = r.json()

    exp = [
        {
            "attributes": "gml_id=<points_ghana_25832.1> id=<1> p_date=<2019-01-01 00:00:00> p_int=<100> p_str=<p_str_1>",
            "geometry": f"POINT EPSG:25832",
            "uid": "a.map.wfs_points_ghana_25832___1"
        },
        {
            "attributes": "gml_id=<points_ghana_25832.2> id=<2> p_date=<2019-01-02 00:00:00> p_int=<200> p_str=<p_str_2>",
            "geometry": f"POINT EPSG:25832",
            "uid": "a.map.wfs_points_ghana_25832___2"
        },
        {
            "attributes": "gml_id=<points_ghana_25832.6> id=<6> p_date=<2019-01-06 00:00:00> p_int=<600> p_str=<p_str_6>",
            "geometry": f"POINT EPSG:25832",
            "uid": "a.map.wfs_points_ghana_25832___6"
        },
        {
            "attributes": "gml_id=<points_ghana_25832.7> id=<7> p_date=<2019-01-07 00:00:00> p_int=<700> p_str=<p_str_7>",
            "geometry": f"POINT EPSG:25832",
            "uid": "a.map.wfs_points_ghana_25832___7"
        }
    ]

    assert u.short_features(r['features']) == exp


def test_find_points_with_reprojection():
    UTM33s = 'EPSG:32733'

    x, y = cc.POINTS.ghana
    x, y = gws.gis.proj.transform_xy(x, y, cc.CRS_3857, UTM33s)
    bbox = (x, y, x + 100, y + 100,)
    bbox = gws.gis.extent.buffer(bbox, _REPROJECTION_ERROR_TOLERANCE)
    shape = gws.gis.shape.from_extent(bbox, UTM33s)

    r = u.cmd('searchFindFeatures', {
        'projectUid': 'a',
        'layerUids': ['a.map.wfs_points_ghana_25832'],
        'crs': UTM33s,
        'resolution': 1,
        'shapes': [shape.props]
    })

    r = r.json()

    exp = [
        {
            "attributes": "gml_id=<points_ghana_25832.1> id=<1> p_date=<2019-01-01 00:00:00> p_int=<100> p_str=<p_str_1>",
            "geometry": "POINT EPSG:32733",
            "uid": "a.map.wfs_points_ghana_25832___1"
        },
        {
            "attributes": "gml_id=<points_ghana_25832.2> id=<2> p_date=<2019-01-02 00:00:00> p_int=<200> p_str=<p_str_2>",
            "geometry": "POINT EPSG:32733",
            "uid": "a.map.wfs_points_ghana_25832___2"
        },
        {
            "attributes": "gml_id=<points_ghana_25832.6> id=<6> p_date=<2019-01-06 00:00:00> p_int=<600> p_str=<p_str_6>",
            "geometry": "POINT EPSG:32733",
            "uid": "a.map.wfs_points_ghana_25832___6"
        },
        {
            "attributes": "gml_id=<points_ghana_25832.7> id=<7> p_date=<2019-01-07 00:00:00> p_int=<700> p_str=<p_str_7>",
            "geometry": "POINT EPSG:32733",
            "uid": "a.map.wfs_points_ghana_25832___7"
        }
    ]

    assert u.short_features(r['features']) == exp


def test_render_squares():
    x, y = cc.POINTS.memphis
    bbox = (x - 300, y, x, y + 300,)

    url = '_/cmd/mapHttpGetBbox/layerUid/a.map.wfs_squares_memphis_25832/bbox/' + gws.as_str_list(bbox)

    r = u.req(url, params={'width': 400, 'height': 400})
    d = u.compare_image_response(r, '/data/squares_400x400.png')
    assert not d
