"""Tests for the SVG draw module."""

import re

import gws
import gws.base.shape
import gws.lib.svg.draw as draw
import gws.lib.crs as crs
import gws.lib.style
import gws.test.util as u


def test_shape_to_fragment_point():
    shape = gws.base.shape.from_wkt('SRID=3857;POINT(100 200)')
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(bounds=bounds, size=[1000, 1000], rotation=0, scale=1, dpi=96)

    # Test with default style
    frg = draw.shape_to_fragment(shape, view)
    assert len(frg) == 1
    assert frg[0].to_string() == '<circle cx="377952" cy="3023622"/>'

    # Test with custom style
    style = _style(with_geometry='all', point_size=20, fill='red', stroke='blue', stroke_width=2)

    frg = draw.shape_to_fragment(shape, view, style=style)
    assert len(frg) == 1
    assert frg[0].to_string() == '<circle cx="377952" cy="3023622" fill="red" stroke="blue" stroke-width="2px" r="10"/>'


def test_shape_to_fragment_linestring():
    """Test converting a linestring shape to SVG frg."""
    shape = gws.base.shape.from_wkt('SRID=3857;LINESTRING(100 100, 200 200, 300 100)')
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(bounds=bounds, size=[1000, 1000], rotation=0, scale=1, dpi=96)

    style = _style(with_geometry='all', stroke='green', stroke_width=3)

    frg = draw.shape_to_fragment(shape, view, style=style)
    assert len(frg) == 1
    assert u.fxml(frg[0].to_string()) == u.fxml("""
        <path d="M 377952.0 3401574.0 L 755905.0 3023622.0 L 1133858.0 3401574.0" fill="none" stroke="green" stroke-width="3px"/>
    """)


def test_shape_to_fragment_polygon():
    """Test converting a polygon shape to SVG frg."""
    shape = gws.base.shape.from_wkt('SRID=3857; POLYGON((100 100, 200 100, 200 200, 100 200, 100 100))')
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(bounds=bounds, size=[1000, 1000], rotation=0, scale=1, dpi=96)

    values = gws.StyleValues(with_geometry='all', fill='yellow', stroke='black', stroke_width=1)
    style = gws.Style()
    style.values = values

    frg = draw.shape_to_fragment(shape, view, style=style)
    assert len(frg) == 1
    xml = frg[0].to_string()
    assert u.fxml(xml) == u.fxml("""
        <path 
            fill-rule="evenodd" 
            d="M 377952.0 3401574.0 L 755905.0 3401574.0 L 755905.0 3023622.0 L 377952.0 3023622.0 L 377952.0 3401574.0 z" 
            fill="yellow" stroke="black" stroke-width="1px"/>
    """)


# def test_shape_to_fragment_with_label():
#     """Test shape with label."""
#     shape = gws.base.shape.from_wkt('SRID=3857; POINT(100 200)')
#     bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
#     view = gws.MapView(
#         bounds=bounds,
#         size=[1000, 1000],
#         rotation=0,
#         scale=1,
#         dpi=96
#     )
#
#     values = gws.StyleValues(
#         with_geometry='all',
#         with_label='all',
#
#         point_size=10,
#         fill='red',
#         label_font_size=12,
#         label_fill='black'
#     )
#     style = gws.Style()
#     style.values = values
#
#     frg = draw.shape_to_fragment(shape, view, label="Test Label", style=style)
#     assert len(frg) == 2  # Point and label
#
#     # Find the label group
#     label_group = None
#     for f in frg:
#         print(f.__dict__)
#     for el in frg:
#         if el.name == 'g' and el.attr('z-index') == 100:
#             label_group = el
#             break
#
#     assert label_group is not None
#
#     # Check text element inside the group
#     text_elements = [c for c in label_group.children() if c.name == 'text']
#     assert len(text_elements) > 0
#
#     # Check tspan with the label text
#     tspans = [c for c in text_elements[0].children() if c.name == 'tspan']
#     assert len(tspans) == 1
#     assert tspans[0].__dict__ == "Test Label"


def test_shape_to_fragment_with_marker():
    """Test shape with marker."""
    shape = gws.base.shape.from_wkt('SRID=3857; LINESTRING(100 100, 200 200)')
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(bounds=bounds, size=[1000, 1000], rotation=0, scale=1, dpi=96)

    style = _style(with_geometry='all', marker='circle', marker_size=16, marker_fill='blue', stroke='black')

    frg = draw.shape_to_fragment(shape, view, style=style)
    assert len(frg) == 2  # Marker and path

    xml = frg[0].to_string() + frg[1].to_string()
    xml = re.sub(r'_M\w+', '_MID', xml)
    assert u.fxml(xml) == u.fxml("""
        <marker id="_MID" viewBox="0 0 16 16" refX="8" refY="8" markerUnits="userSpaceOnUse" markerWidth="16" markerHeight="16">
            <circle fill="blue" cx="8" cy="8" r="8"/>
        </marker>
        <path d="M 377952.0 3401574.0 L 755905.0 3023622.0" fill="none" stroke="black" stroke-width="1px" 
        marker-start="url(#_MID)" marker-mid="url(#_MID)" marker-end="url(#_MID)"/>
    """)


def test_soup_to_fragment():
    """Test converting a soup to SVG frg."""
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(bounds=bounds, size=[1000, 1000], rotation=0, scale=1, dpi=96)

    points = [(100, 100), (200, 200), (300, 100)]
    tags = [
        ('line', {'x1': ['x', 0], 'y1': ['y', 0], 'x2': ['x', 1], 'y2': ['y', 1], 'stroke': 'black'}),
        ('text', {'x': ['x', 2], 'y': ['y', 2], 'transform': ['r', 0, 1, 2]}, 'Test Text'),
    ]

    frg = draw.soup_to_fragment(view, points, tags)
    assert len(frg) == 2
    xml = frg[0].to_string() + frg[1].to_string()
    assert u.fxml(xml) == u.fxml("""
        <line x1="377952" y1="3401574" x2="755905" y2="3023622" stroke="black"/>
        <text x="1133858" y="3401574" transform="rotate(-45, 1133858, 3401574)">
            Test Text
        </text>
    """)


def test_empty_shape():
    """Test handling of empty shapes."""
    shape = gws.base.shape.from_wkt('SRID=3857;POLYGON EMPTY')
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(bounds=bounds, size=[1000, 1000], rotation=0, scale=1, dpi=96)

    frg = draw.shape_to_fragment(shape, view)
    assert len(frg) == 0


def test_label_visibility():
    """Test label visibility based on scale."""
    shape = gws.base.shape.from_wkt('SRID=3857;POINT(100 200)')

    # Create view with scale 1:1000
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(bounds=bounds, size=[1000, 1000], rotation=0, scale=1000, dpi=96)

    # Style with label visible only at scales between 1:500 and 1:2000

    style = _style(
        with_geometry='all',
        with_label='all',
        label_min_scale=500,
        label_max_scale=2000,
        point_size=10,
        fill='red',
        label_font_size=12,
        label_fill='black',
    )

    # Label should be visible at scale 1:1000
    frg = draw.shape_to_fragment(shape, view, label='Test Label', style=style)
    assert len(frg) == 2  # Point and label

    # Create view with scale 1:3000 (outside max_scale)
    view.scale = 3000
    frg = draw.shape_to_fragment(shape, view, label='Test Label', style=style)
    assert len(frg) == 1  # Only point, no label

    # Create view with scale 1:100 (outside min_scale)
    view.scale = 100
    frg = draw.shape_to_fragment(shape, view, label='Test Label', style=style)
    assert len(frg) == 1  # Only point, no label


def test_multigeometry():
    """Test handling of multi-geometries."""
    shape = gws.base.shape.from_wkt('SRID=3857;MULTIPOINT((100 100), (200 200))')
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(bounds=bounds, size=[1000, 1000], rotation=0, scale=1, dpi=96)

    values = gws.StyleValues(with_geometry='all', point_size=10, fill='red')
    style = gws.Style()
    style.values = values

    frg = draw.shape_to_fragment(shape, view, style=style)
    assert len(frg) == 1
    xml = frg[0].to_string()
    assert u.fxml(xml) == u.fxml("""
        <g>
            <circle cx="377952" cy="3401574" fill="red" r="5"/>
            <circle cx="755905" cy="3023622" fill="red" r="5"/>
        </g>
    """)


def _style(**kwargs) -> gws.Style:
    return gws.lib.style.Object('', '', gws.StyleValues(**kwargs))
