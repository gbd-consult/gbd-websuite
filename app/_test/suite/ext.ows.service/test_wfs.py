import gws.gis.shape
import gws.gis.proj
import gws.tools.json2
import gws.gis.extent

import _test.util as u
import _test.common.const as cc


def test_get_capabilities():
    r = u.req('_/cmd/owsHttpGetWfs', params={
        'projectUid': 'wfs1',
        'SERVICE': 'WFS',
        'REQUEST': 'GetCapabilities'
    })

    assert u.xml(r) == u.xml('/data/response_xml/wfs_GetCapabilities_wfs1.xml')

def test_get_describefeaturetype():
    r = u.req('_/cmd/owsHttpGetWfs', params={
        'projectUid': 'wfs1',
        'SERVICE': 'WFS',
        'REQUEST': 'DescribeFeatureType',
    })

    assert u.xml(r) == u.xml('/data/response_xml/wfs_DescribeFeatureType_wfs1.xml')


def test_get_all_features():
    x, y = cc.POINTS.paris

    r = u.req('_/cmd/owsHttpGetWfs', params={
        'projectUid': 'wfs1',
        'SERVICE': 'WFS',
        'REQUEST': 'GetFeature',
        'TYPENAMES': 'paris_3857'
    })

    assert u.xml(r.text).count('<wfs:member>') == 50


def test_get_features_with_bbox():
    x, y = cc.POINTS.paris

    r = u.req('_/cmd/owsHttpGetWfs', params={
        'projectUid': 'wfs1',
        'SERVICE': 'WFS',
        'REQUEST': 'GetFeature',
        'TYPENAMES': 'paris_3857',
        'BBOX': u.strlist([x, y, x + 150, y + 50]),
    })

    exp = """
        <wfs:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2" xmlns:gws="http://gws.gbd-consult.de" xmlns:wfs="http://www.opengis.net/wfs/2.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="2.0" xsi:schemaLocation="http://www.opengis.net/gml/3.2 http://schemas.opengis.net/gml/3.2.1/gml.xsd http://www.opengis.net/wfs/2.0 http://schemas.opengis.net/wfs/2.0/wfs.xsd">
            <wfs:member>
                <gws:paris_3857 gml:id="1">
                    <gws:id>1</gws:id>
                    <gws:p_str>paris_3857/1</gws:p_str>
                    <gws:p_int>100</gws:p_int>
                    <gws:p_date>2019-01-01</gws:p_date>
                    <gws:geometry>
                        <gml:Point srsName="urn:ogc:def:crs:EPSG::3857">
                            <gml:pos srsDimension="2">254451.84 6250716.48</gml:pos>
                        </gml:Point>
                    </gws:geometry>
                </gws:paris_3857>
            </wfs:member>
            <wfs:member>
                <gws:paris_3857 gml:id="2">
                    <gws:id>2</gws:id>
                    <gws:p_str>paris_3857/2</gws:p_str>
                    <gws:p_int>200</gws:p_int>
                    <gws:p_date>2019-01-02</gws:p_date>
                    <gws:geometry>
                        <gml:Point srsName="urn:ogc:def:crs:EPSG::3857">
                            <gml:pos srsDimension="2">254551.84 6250716.48</gml:pos>
                        </gml:Point>
                    </gws:geometry>
                </gws:paris_3857>
            </wfs:member>
        </wfs:FeatureCollection>
    """

    assert u.xml(r) == u.xml(exp)
