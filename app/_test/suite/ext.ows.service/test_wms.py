import gws.gis.shape
import gws.gis.proj
import gws.tools.json2
import gws.gis.extent

import _test.util as u
import _test.common.const as cc


def test_get_capabilities():
    r = u.req('_/cmd/owsHttpService/uid/wms', params={
        'projectUid': 'a',
        'SERVICE': 'WMS',
        'REQUEST': 'GetCapabilities'
    })
    a, b = u.compare_xml(r, path='/data/response_xml/wms_GetCapabilities_wms1.xml')
    assert a == b


def test_get_inspire_capabilities():
    r = u.req('_/cmd/owsHttpService/uid/wms_inspire', params={
        'projectUid': 'a',
        'SERVICE': 'WMS',
        'REQUEST': 'GetCapabilities'
    })
    a, b = u.compare_xml(r, path='/data/response_xml/wms_GetCapabilities_wms_inspire.xml')
    assert a == b


def test_get_map():
    x, y = cc.POINTS.paris

    r = u.req('_/cmd/owsHttpService/uid/wms', params={
        'projectUid': 'a',
        'SERVICE': 'WMS',
        'REQUEST': 'GetMap',
        'LAYERS': 'paris_3857',
        'BBOX': u.strlist([x, y, x + 350, y + 350]),
        'WIDTH': 400,
        'HEIGHT': 400,
    })
    a, b = u.compare_image(r, path='/data/response_images/wms_paris_3857_400x400.png')
    assert a == b


# points.paris + x=300, y=200

_paris_point_14 = """
    <wfs:FeatureCollection 
        timeStamp="..." 
        numberMatched="1" numberReturned="1" 
        xmlns:aaa="http://ns-aaa"
        xmlns:gml="http://www.opengis.net/gml/3.2" 
        xmlns:wfs="http://www.opengis.net/wfs/2.0"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://ns-aaa http://ns-aaa-schema http://www.opengis.net/gml/3.2 http://schemas.opengis.net/gml/3.2.1/gml.xsd http://www.opengis.net/wfs/2.0 http://schemas.opengis.net/wfs/2.0/wfs.xsd">

        <wfs:member>
            <aaa:paris gml:id="14">
                <aaa:id>14</aaa:id>
                <aaa:p_str>paris_3857/14</aaa:p_str>
                <aaa:p_int>1400</aaa:p_int>
                <aaa:p_date>2019-01-14</aaa:p_date>
                <aaa:geometry>
                    <gml:Point srsName="urn:ogc:def:crs:EPSG::3857">
                        <gml:pos srsDimension="2">254751.84 6250916.48</gml:pos>
                    </gml:Point>
                </aaa:geometry>
            </aaa:paris>
        </wfs:member>
    </wfs:FeatureCollection>
"""


def test_get_features():
    x, y = cc.POINTS.paris

    r = u.req('_/cmd/owsHttpService/uid/wms', params={
        'projectUid': 'a',
        'SERVICE': 'WMS',
        'REQUEST': 'GetFeatureInfo',
        'QUERY_LAYERS': 'paris_3857',
        'BBOX': u.strlist([x, y, x + 350, y + 350]),
        'WIDTH': 350,
        'HEIGHT': 350,
        'I': 300,
        'J': 150,
    })

    a, b = u.compare_xml(r, _paris_point_14)
    assert a == b


def test_get_features_with_resolution():
    x, y = cc.POINTS.paris

    r = u.req('_/cmd/owsHttpService/uid/wms', params={
        'projectUid': 'a',
        'SERVICE': 'WMS',
        'REQUEST': 'GetFeatureInfo',
        'QUERY_LAYERS': 'paris_3857',
        'BBOX': u.strlist([x, y, x + 350, y + 350]),
        'WIDTH': 350 * 3,
        'HEIGHT': 350 * 3,
        'I': 300 * 3,
        'J': 150 * 3,
    })

    a, b = u.compare_xml(r, _paris_point_14)
    assert a == b


def test_get_features_with_reprojection():
    x, y = cc.POINTS.paris

    x, y = gws.gis.proj.transform_xy(x + 300, y + 200, cc.CRS_3857, cc.CRS_25832)

    r = u.req('_/cmd/owsHttpService/uid/wms', params={
        'projectUid': 'a',
        'SERVICE': 'WMS',
        'REQUEST': 'GetFeatureInfo',
        'QUERY_LAYERS': 'paris_3857',
        'BBOX': u.strlist([x - 20, y - 20, x + 20, y + 20]),
        'WIDTH': 40,
        'HEIGHT': 40,
        'CRS': 'EPSG:25832',
        'I': 20,
        'J': 20,
    })

    a, b = u.compare_xml(r, _paris_point_14)
    assert a == b
