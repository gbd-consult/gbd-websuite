import gws.gis.shape
import gws.gis.proj
import gws.tools.json2
import gws.gis.extent

import _test.util as u
import _test.common.const as cc


def test_get_capabilities():
    r = u.req('_/cmd/owsHttp', params={
        'projectUid': 'wms1',
        'serviceName': 'wms',
        'SERVICE': 'WMS',
        'REQUEST': 'GetCapabilities'
    })

    assert u.xml(r.text) == u.read('/data/wms_GetCapabilities_wms1.xml')


def test_get_map():
    x, y = cc.POINTS.paris

    r = u.req('_/cmd/owsHttp', params={
        'projectUid': 'wms1',
        'serviceName': 'wms',
        'SERVICE': 'WMS',
        'REQUEST': 'GetMap',
        'LAYERS': 'paris_3857',
        'BBOX': u.strlist([x, y, x + 350, y + 350]),
        'WIDTH': 400,
        'HEIGHT': 400,
    })

    d = u.compare_image_response(r, '/data/wms_paris_3857_400x400.png')
    assert not d


def test_get_features1():
    x, y = cc.POINTS.paris

    r = u.req('_/cmd/owsHttp', params={
        'projectUid': 'wms1',
        'serviceName': 'wms',
        'SERVICE': 'WMS',
        'REQUEST': 'GetFeatureInfo',
        'QUERY_LAYERS': 'paris_3857',
        'BBOX': u.strlist([x, y, x + 350, y + 350]),
        'WIDTH': 350,
        'HEIGHT': 350,
        # should give us point 14 (x + 300, y + 350 - 150)
        'I': 300,
        'J': 150,
    })

    exp = """
        <wfs:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2" xmlns:gws="http://gws.gbd-consult.de" xmlns:wfs="http://www.opengis.net/wfs/2.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="2.0" xsi:schemaLocation="http://www.opengis.net/gml/3.2 http://schemas.opengis.net/gml/3.2.1/gml.xsd http://www.opengis.net/wfs/2.0 http://schemas.opengis.net/wfs/2.0/wfs.xsd">
            <wfs:member>
                <gws:paris_3857 gml:id="14">
                    <gws:id>14</gws:id>
                    <gws:p_str>paris_3857/14</gws:p_str>
                    <gws:p_int>1400</gws:p_int>
                    <gws:p_date>2019-01-14</gws:p_date>
                    <gws:geometry>
                        <gml:Point srsName="urn:ogc:def:crs:EPSG::3857">
                            <gml:pos srsDimension="2">254751.84 6250916.48</gml:pos>
                        </gml:Point>
                    </gws:geometry>
                </gws:paris_3857>
            </wfs:member>
        </wfs:FeatureCollection>
    """

    assert u.xml(r.text) == u.xml(exp)

def test_get_features_with_resolution():
    x, y = cc.POINTS.paris

    r = u.req('_/cmd/owsHttp', params={
        'projectUid': 'wms1',
        'serviceName': 'wms',
        'SERVICE': 'WMS',
        'REQUEST': 'GetFeatureInfo',
        'QUERY_LAYERS': 'paris_3857',
        'BBOX': u.strlist([x, y, x + 350, y + 350]),
        'WIDTH': 350 * 3,
        'HEIGHT': 350 * 3,
        # should give us point 14 (x + 300, y + 350 - 150)
        'I': 300 * 3,
        'J': 150 * 3,
    })

    exp = """
        <wfs:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2" xmlns:gws="http://gws.gbd-consult.de" xmlns:wfs="http://www.opengis.net/wfs/2.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="2.0" xsi:schemaLocation="http://www.opengis.net/gml/3.2 http://schemas.opengis.net/gml/3.2.1/gml.xsd http://www.opengis.net/wfs/2.0 http://schemas.opengis.net/wfs/2.0/wfs.xsd">
            <wfs:member>
                <gws:paris_3857 gml:id="14">
                    <gws:id>14</gws:id>
                    <gws:p_str>paris_3857/14</gws:p_str>
                    <gws:p_int>1400</gws:p_int>
                    <gws:p_date>2019-01-14</gws:p_date>
                    <gws:geometry>
                        <gml:Point srsName="urn:ogc:def:crs:EPSG::3857">
                            <gml:pos srsDimension="2">254751.84 6250916.48</gml:pos>
                        </gml:Point>
                    </gws:geometry>
                </gws:paris_3857>
            </wfs:member>
        </wfs:FeatureCollection>
    """

    assert u.xml(r.text) == u.xml(exp)

def test_get_features_with_reprojection():
    x, y = cc.POINTS.dus

    x, y = gws.gis.proj.transform_xy(x, y, cc.CRS_3857, cc.CRS_25832)
    bbox = gws.gis.extent.transform((x, y, x + 350, y + 350), cc.CRS_25832, cc.CRS_3857)

    r = u.req('_/cmd/owsHttp', params={
        'projectUid': 'wms1',
        'serviceName': 'wms',
        'SERVICE': 'WMS',
        'REQUEST': 'GetFeatureInfo',
        'QUERY_LAYERS': 'dus_25832',
        'BBOX': u.strlist(bbox),
        'WIDTH': 350,
        'HEIGHT': 350,
        # should give us square 14 (x + 300, y + 350 - 150)
        'I': 300,
        'J': 150,
    })

    exp = """
        <wfs:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2" xmlns:gws="http://gws.gbd-consult.de" xmlns:wfs="http://www.opengis.net/wfs/2.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="2.0" xsi:schemaLocation="http://www.opengis.net/gml/3.2 http://schemas.opengis.net/gml/3.2.1/gml.xsd http://www.opengis.net/wfs/2.0 http://schemas.opengis.net/wfs/2.0/wfs.xsd">
            <wfs:member>
                <gws:dus_25832 gml:id="14">
                    <gws:id>14</gws:id>
                    <gws:p_str>dus_25832/14</gws:p_str>
                    <gws:p_int>1400</gws:p_int>
                    <gws:p_date>2019-01-14</gws:p_date>
                    <gws:geometry>
                        <gml:Polygon srsName="urn:ogc:def:crs:EPSG::25832">
                            <gml:exterior>
                                <gml:LinearRing>
                                    <gml:posList srsDimension="2">344708.76 5677207.8 344758.76 5677207.8 344758.76 5677257.8 344708.76 5677257.8 344708.76 5677207.8</gml:posList>
                                </gml:LinearRing>
                            </gml:exterior>
                        </gml:Polygon>
                    </gws:geometry>
                </gws:dus_25832>
            </wfs:member>
        </wfs:FeatureCollection>
    """

    assert u.xml(r.text) == u.xml(exp)
