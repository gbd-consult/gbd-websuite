"""Tests for the writer module"""

import gws
import gws.base.shape
import gws.lib.crs
import gws.lib.gml.writer as writer
import gws.lib.xmlx
import gws.test.util as u


def test_shape_to_element():
    p = gws.base.shape.from_xy(12.34567, 5.6789, crs=gws.lib.crs.WEBMERCATOR)

    xml = writer.shape_to_element(p).to_string()
    assert xml == u.fxml("""
        <gml:Point srsName="urn:ogc:def:crs:EPSG::3857">
            <gml:pos srsDimension="2">12.35 5.68</gml:pos>
        </gml:Point>
    """)


def test_shape_to_element_crs_format():
    p = gws.base.shape.from_xy(12.34567, 5.6789, crs=gws.lib.crs.WEBMERCATOR)

    xml = writer.shape_to_element(p, crs_format=gws.CrsFormat.epsg).to_string()
    assert xml == u.fxml("""
        <gml:Point srsName="EPSG:3857">
            <gml:pos srsDimension="2">12.35 5.68</gml:pos>
        </gml:Point>
    """)


def test_shape_to_element_coordinate_precision():
    p = gws.base.shape.from_xy(12.34567, 5.6789, crs=gws.lib.crs.WEBMERCATOR)
    xml = writer.shape_to_element(p, coordinate_precision=4).to_string()
    assert xml == u.fxml("""
        <gml:Point srsName="urn:ogc:def:crs:EPSG::3857">
            <gml:pos srsDimension="2">12.3457 5.6789</gml:pos>
        </gml:Point>
    """)


def test_shape_to_element_xy():
    p = gws.base.shape.from_xy(12.34567, 5.6789, crs=gws.lib.crs.WGS84)

    xml = writer.shape_to_element(p, always_xy=True).to_string()
    assert xml == u.fxml("""
        <gml:Point srsName="urn:ogc:def:crs:EPSG::4326">
            <gml:pos srsDimension="2">12.34567 5.6789</gml:pos>
        </gml:Point>
    """)

    xml = writer.shape_to_element(p, always_xy=False).to_string()
    assert xml == u.fxml("""
        <gml:Point srsName="urn:ogc:def:crs:EPSG::4326">
            <gml:pos srsDimension="2">5.6789 12.34567</gml:pos>
        </gml:Point>
    """)


def test_shape_to_element_namespace():
    p = gws.base.shape.from_xy(12.34567, 5.6789, crs=gws.lib.crs.WEBMERCATOR)
    ns = gws.lib.xmlx.namespace.get('wms')
    xml = writer.shape_to_element(p, namespace=ns).to_string()
    assert xml == u.fxml("""
        <wms:Point srsName="urn:ogc:def:crs:EPSG::3857">
            <wms:pos srsDimension="2">12.35 5.68</wms:pos>
        </wms:Point>
    """)


def test_shape_to_element_with_xmlns():
    p = gws.base.shape.from_xy(12.34567, 5.6789, crs=gws.lib.crs.WEBMERCATOR)
    xml = writer.shape_to_element(p, with_xmlns=False).to_string()
    assert xml == u.fxml("""
        <Point srsName="urn:ogc:def:crs:EPSG::3857">
            <pos srsDimension="2">12.35 5.68</pos>
        </Point>
    """)


def test_shape_to_element_with_inline_xmlns():
    p = gws.base.shape.from_xy(12.34567, 5.6789, crs=gws.lib.crs.WEBMERCATOR)
    xml = writer.shape_to_element(p, with_inline_xmlns=True).to_string()
    assert xml == u.fxml("""
        <gml:Point srsName="urn:ogc:def:crs:EPSG::3857" xmlns:gml="http://www.opengis.net/gml">
            <gml:pos srsDimension="2" xmlns:gml="http://www.opengis.net/gml">12.35 5.68</gml:pos>
        </gml:Point>
    """)
