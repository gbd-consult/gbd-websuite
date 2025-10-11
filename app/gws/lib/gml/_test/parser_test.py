"""Tests for the parser module"""

import gws
import gws.lib.crs
import gws.lib.gml.parser as parser
import gws.test.util as u
import gws.lib.xmlx


def test_parse_envelope_wgs():
    box = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Box>
                <gml:coordinates>1,2 3,4</gml:coordinates>
            </gml:Box>
        </root>
    """)
    wgs = gws.lib.crs.WGS84
    p = parser.parse_envelope(box.findfirst(), wgs, False)
    assert p.crs == wgs
    assert p.extent == (2.0, 1.0, 4.0, 3.0)


def test_parse_envelope():
    box = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Box srsName="EPSG:3857">
                <gml:coordinates>1,2 3,4</gml:coordinates>
            </gml:Box>
        </root>
    """)
    wm = gws.lib.crs.WEBMERCATOR
    p = parser.parse_envelope(box.findfirst())
    assert p.crs == wm
    assert p.extent == (1.0, 2.0, 3.0, 4.0)


def test_parse_envelope_always_xy():
    box = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Envelope srsDimension="2">
                <gml:lowerCorner>1 2</gml:lowerCorner>
                <gml:upperCorner>3 4</gml:upperCorner>
            </gml:Envelope>
        </root>
    """)
    wgs = gws.lib.crs.WGS84

    p = parser.parse_envelope(box.findfirst(), wgs)
    assert p.crs == wgs
    assert p.extent == (2.0, 1.0, 4.0, 3.0)


def test_parse_envelope_gml3():
    box = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Envelope srsDimension="2" srsName="EPSG:3857">
                <gml:lowerCorner>1 2</gml:lowerCorner>
                <gml:upperCorner>3 4</gml:upperCorner>
            </gml:Envelope>
        </root>
    """)
    wm = gws.lib.crs.WEBMERCATOR
    p = parser.parse_envelope(box.findfirst())
    assert p.crs == wm
    assert p.extent == (1.0, 2.0, 3.0, 4.0)


def test_parse_envelope_no_crs():
    box = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Box>
                <gml:coordinates>1,2 3,4</gml:coordinates>
            </gml:Box>
        </root>
    """)
    with u.raises(Exception):
        parser.parse_envelope(box.findfirst())


def test_parse_envelope_no_envelope():
    box = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Foo srsName="EPSG:3857">
                <gml:coordinates>1,2 3,4</gml:coordinates>
            </gml:Foo>
        </root>
    """)
    with u.raises(Exception):
        parser.parse_envelope(box.findfirst())


def test_is_geometry_element():
    element = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Point>
                <gml:coordinates>100,200</gml:coordinates>
            </gml:Point>
        </root>
    """)
    assert parser.is_geometry_element(element.findfirst())


def test_is_not_geometry_element():
    element = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:point>
                <gml:coordinates>100,200</gml:coordinates>
            </gml:point>
        </root>
    """)
    assert not parser.is_geometry_element(element)


##########
def test_parse_shape():
    shape = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Point srsName="EPSG:3857">
                <gml:coordinates>100,200</gml:coordinates>
            </gml:Point>
        </root>
    """)
    p = parser.parse_shape(shape.findfirst())
    assert p.x == 100 and p.y == 200


def test_parse_shape_no_crs():
    shape = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Point>
                <gml:coordinates>100,200</gml:coordinates>
            </gml:Point>
        </root>
    """)
    with u.raises(Exception):
        parser.parse_shape(shape.findfirst())


def test_parse_shape_wgs():
    shape = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Point>
                <gml:coordinates>100,200</gml:coordinates>
            </gml:Point>
        </root>
    """)
    wgs = gws.lib.crs.WGS84
    p =  parser.parse_shape(shape.findfirst(), wgs)
    assert p.x == 200 and p.y == 100


def test_parse_shape_always_xy():
    shape = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Point>
                <gml:coordinates>100,200</gml:coordinates>
            </gml:Point>
        </root>
    """)
    wgs = gws.lib.crs.WGS84
    p = parser.parse_shape(shape.findfirst(), wgs, always_xy=True)
    assert p.x == 100 and p.y == 200


def test_parse_geometry():
    shape = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Polygon>
                <gml:outerBoundaryIs>
                    <gml:LinearRing>
                        <gml:coordinates>0,0 100,0 100,100 0,100 0,0</gml:coordinates>
                    </gml:LinearRing>
                </gml:outerBoundaryIs>
            </gml:Polygon>
        </root>
    """)
    assert parser.parse_geometry(shape.findfirst()) == {
        'coordinates': [
            [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [0.0, 0.0]],
        ],
        'type': 'Polygon',
    }


def test_parse_geometry_unknown_type():
    shape = gws.lib.xmlx.from_string("""
        <root xmlns:gml="http://www.opengis.net/gml">
            <gml:Points>
                <gml:pointMember>
                    <gml:Point>
                        <gml:coordinates>100.0, 0.0</gml:coordinates>
                    </gml:Point>
                </gml:pointMember>
                <gml:pointMember>
                    <gml:Point>
                        <gml:coordinates>101.0, 1.0</gml:coordinates>
                    </gml:Point>
                </gml:pointMember>
            </gml:Points>
        </root>
    """)
    with u.raises(Exception):
        parser.parse_geometry(shape.findfirst())
