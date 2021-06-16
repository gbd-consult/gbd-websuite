import gws.gis.shape
import gws.gis.proj
import gws.lib.json2
import gws.gis.extent

import _test.util as u
import _test.common.const as cc


def test_get_capabilities():
    r = u.req('_/cmd/owsHttpService/uid/wfs', params={
        'projectUid': 'a',
        'SERVICE': 'WFS',
        'REQUEST': 'GetCapabilities'
    })

    a, b = u.compare_xml(r, path='/data/response_xml/wfs_GetCapabilities_wfs1.xml')
    assert a == b

def test_get_describefeaturetype():
    r = u.req('_/cmd/owsHttpService/uid/wfs', params={
        'projectUid': 'a',
        'SERVICE': 'WFS',
        'REQUEST': 'DescribeFeatureType',
    })

    a, b = u.compare_xml(r, path='/data/response_xml/wfs_DescribeFeatureType_wfs1.xml')
    assert a == b


def test_get_all_features():
    x, y = cc.POINTS.paris

    r = u.req('_/cmd/owsHttpService/uid/wfs', params={
        'projectUid': 'a',
        'SERVICE': 'WFS',
        'REQUEST': 'GetFeature',
        'TYPENAMES': 'paris_3857'
    })

    assert u.xml(r.text).count('<wfs:member>') == 50


def test_get_features_with_bbox():
    x, y = cc.POINTS.paris

    r = u.req('_/cmd/owsHttpService/uid/wfs', params={
        'projectUid': 'a',
        'SERVICE': 'WFS',
        'REQUEST': 'GetFeature',
        'TYPENAMES': 'paris_3857',
        'BBOX': u.strlist([x, y, x + 150, y + 50]),
    })

    exp = """
        <wfs:FeatureCollection timeStamp="..." 
                               numberMatched="2" numberReturned="2" 
                               xmlns:aaa="http://ns-aaa"
                               xmlns:gml="http://www.opengis.net/gml/3.2" 
                               xmlns:wfs="http://www.opengis.net/wfs/2.0"
                               xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                               xsi:schemaLocation="http://ns-aaa http://ns-aaa-schema http://www.opengis.net/gml/3.2 http://schemas.opengis.net/gml/3.2.1/gml.xsd http://www.opengis.net/wfs/2.0 http://schemas.opengis.net/wfs/2.0/wfs.xsd">
            <wfs:member>
                <aaa:paris gml:id="1">
                    <aaa:id>1</aaa:id>
                    <aaa:p_str>paris_3857/1</aaa:p_str>
                    <aaa:p_int>100</aaa:p_int>
                    <aaa:p_date>2019-01-01</aaa:p_date>
                    <aaa:geometry>
                        <gml:Point srsName="urn:ogc:def:crs:EPSG::3857">
                            <gml:pos srsDimension="2">254451.84 6250716.48</gml:pos>
                        </gml:Point>
                    </aaa:geometry>
                </aaa:paris>
            </wfs:member>
            <wfs:member>
                <aaa:paris gml:id="2">
                    <aaa:id>2</aaa:id>
                    <aaa:p_str>paris_3857/2</aaa:p_str>
                    <aaa:p_int>200</aaa:p_int>
                    <aaa:p_date>2019-01-02</aaa:p_date>
                    <aaa:geometry>
                        <gml:Point srsName="urn:ogc:def:crs:EPSG::3857">
                            <gml:pos srsDimension="2">254551.84 6250716.48</gml:pos>
                        </gml:Point>
                    </aaa:geometry>
                </aaa:paris>
            </wfs:member>
        </wfs:FeatureCollection>
    """

    a, b = u.compare_xml(r, exp)
    assert a == b
