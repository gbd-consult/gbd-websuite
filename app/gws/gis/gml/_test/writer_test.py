"""Tests for the writer module"""

import gws
import gws.test.util as u

import gws.gis.gml.writer as writer
import gws.gis.gml.parser as parser
import gws.lib.xmlx


def test_shape_to_element():
    gml = gws.lib.xmlx.from_string('''    <root xmlns:gml="http://www.opengis.net/gml">
                                                    <gml:Point srsName = "EPSG:3857">
                                                        <gml:coordinates>100.12345,200.12345</gml:coordinates>
                                                    </gml:Point>
                                                </root>''')
    shape = parser.parse_shape(gml.findfirst())
    assert writer.shape_to_element(shape).to_string() == u.fxml("""<gml:Point srsName="urn:ogc:def:crs:EPSG::3857">
                                                                        <gml:pos srsDimension="2">100.12 200.12</gml:pos>
                                                                     </gml:Point>""")


def test_shape_to_element_coordinate_precision():
    gml = gws.lib.xmlx.from_string('''    <root xmlns:gml="http://www.opengis.net/gml">
                                                    <gml:Point srsName = "EPSG:3857">
                                                        <gml:coordinates>100.12345,200.12345</gml:coordinates>
                                                    </gml:Point>
                                                </root>''')
    shape = parser.parse_shape(gml.findfirst())
    assert writer.shape_to_element(shape, coordinate_precision=4).to_string() == u.fxml("""<gml:Point srsName="urn:ogc:def:crs:EPSG::3857">
                                                                        <gml:pos srsDimension="2">100.1235 200.1234</gml:pos>
                                                                     </gml:Point>""")


def test_shape_to_element_xy():
    gml = gws.lib.xmlx.from_string('''    <root xmlns:gml="http://www.opengis.net/gml">
                                                    <gml:Point srsName="EPSG:4326">
                                                        <gml:coordinates>100.12345,200.12345</gml:coordinates>
                                                    </gml:Point>
                                                </root>''')
    shape = parser.parse_shape(gml.findfirst())
    assert writer.shape_to_element(shape, always_xy=True).to_string() == u.fxml("""<gml:Point srsName="urn:ogc:def:crs:EPSG::4326">
                                                                        <gml:pos srsDimension="2">200.12345 100.12345</gml:pos>
                                                                     </gml:Point>""")


def test_shape_to_element_crs_format():
    gml = gws.lib.xmlx.from_string('''    <root xmlns:gml="http://www.opengis.net/gml">
                                                       <gml:Point srsName = "EPSG:3857">
                                                           <gml:coordinates>100.12345,200.12345</gml:coordinates>
                                                       </gml:Point>
                                                   </root>''')
    shape = parser.parse_shape(gml.findfirst())
    crs_format = gws.CrsFormat.epsg
    assert writer.shape_to_element(shape, crs_format=crs_format).to_string() == u.fxml("""<gml:Point srsName="EPSG:3857">
                                                                           <gml:pos srsDimension="2">100.12 200.12</gml:pos>
                                                                        </gml:Point>""")


def test_shape_to_element_namespace():
    gml = gws.lib.xmlx.from_string('''    <root xmlns:gml="http://www.opengis.net/gml">
                                                        <gml:Point srsName = "EPSG:3857">
                                                            <gml:coordinates>100.12345,200.12345</gml:coordinates>
                                                        </gml:Point>
                                                    </root>''')
    shape = parser.parse_shape(gml.findfirst())
    ns = gws.lib.xmlx.namespace.get('wms')
    assert writer.shape_to_element(shape, namespace=ns).to_string() == u.fxml("""<wms:Point srsName="urn:ogc:def:crs:EPSG::3857">
                                                                            <wms:pos srsDimension="2">100.12 200.12</wms:pos>
                                                                         </wms:Point>""")


def test_shape_to_element_with_xmlns():
    gml = gws.lib.xmlx.from_string('''    <root xmlns:gml="http://www.opengis.net/gml">
                                                    <gml:Point srsName = "EPSG:3857">
                                                        <gml:coordinates>100.12345,200.12345</gml:coordinates>
                                                    </gml:Point>
                                                </root>''')
    shape = parser.parse_shape(gml.findfirst())
    assert writer.shape_to_element(shape, with_xmlns=False).to_string() == u.fxml("""<Point srsName="urn:ogc:def:crs:EPSG::3857">
                                                                        <pos srsDimension="2">100.12 200.12</pos>
                                                                     </Point>""")


def test_shape_to_element_with_inline_xmlns():
    gml = gws.lib.xmlx.from_string('''    <root xmlns:gml="http://www.opengis.net/gml">
                                                    <gml:Point srsName = "EPSG:3857">
                                                        <gml:coordinates>100.12345,200.12345</gml:coordinates>
                                                    </gml:Point>
                                                </root>''')
    shape = parser.parse_shape(gml.findfirst())
    assert writer.shape_to_element(shape, with_inline_xmlns=True).to_string() == u.fxml("""<gml:Point srsName="urn:ogc:def:crs:EPSG::3857" xmlns:gml="http://www.opengis.net/gml">
                                                                        <gml:pos srsDimension="2">100.12 200.12</gml:pos>
                                                                     </gml:Point>""")
