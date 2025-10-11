"""Tests for the SVG draw module."""

import gws
import gws.base.shape
import gws.lib.svg.draw as draw
import gws.lib.crs as crs
import gws.test.util as u


def test_shape_to_fragment_point():
    """Test converting a point shape to SVG fragment."""
    shape = gws.base.shape.from_wkt('SRID=4326;POINT(100 200)')
    bounds = gws.Bounds(crs=crs.WGS84,extent=[0, 0, 1000, 1000])
    view = gws.MapView(
        bounds=bounds,
        size=[1000, 1000],
        rotation=0,
        scale=1,
        dpi=96
    )
    
    # Test with default style
    fragment = draw.shape_to_fragment(shape, view)
    assert len(fragment) == 1
    assert fragment[0].name == 'circle'
    assert fragment[0].attr('cx') == '377952'
    assert fragment[0].attr('cy') == '3023622'
    
    # Test with custom style
    values = gws.StyleValues(
        with_geometry='all',
        point_size=20,
        fill='red',
        stroke='blue',
        stroke_width=2)
    style = gws.Style()
    style.values = values

    fragment = draw.shape_to_fragment(shape, view, style=style)
    assert len(fragment) == 1
    assert fragment[0].name == 'circle'
    assert fragment[0].attr('cx') == '377952'
    assert fragment[0].attr('cy') == '3023622'
    assert fragment[0].attr('r') == '10'  # point_size/2
    assert fragment[0].attr('fill') == 'red'
    assert fragment[0].attr('stroke') == 'blue'
    assert fragment[0].attr('stroke-width') == '2px'


def test_shape_to_fragment_linestring():
    """Test converting a linestring shape to SVG fragment."""
    shape = gws.base.shape.from_wkt('SRID=4326;LINESTRING(100 100, 200 200, 300 100)')
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(
        bounds=bounds,
        size=[1000, 1000],
        rotation=0,
        scale=1,
        dpi=96
    )

    values = gws.StyleValues(
        with_geometry='all',
        stroke='green',
        stroke_width=3
    )
    style = gws.Style()
    style.values = values
    
    fragment = draw.shape_to_fragment(shape, view, style=style)
    assert len(fragment) == 1
    assert fragment[0].name == 'path'
    assert fragment[0].attr('d') == 'M 377952.0 3401574.0 L 755905.0 3023622.0 L 1133858.0 3401574.0'
    assert fragment[0].attr('stroke') == 'green'
    assert fragment[0].attr('stroke-width') == '3px'
    assert fragment[0].attr('fill') == 'none'


def test_shape_to_fragment_polygon():
    """Test converting a polygon shape to SVG fragment."""
    shape = gws.base.shape.from_wkt('SRID=4326; POLYGON((100 100, 200 100, 200 200, 100 200, 100 100))')
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(
        bounds=bounds,
        size=[1000, 1000],
        rotation=0,
        scale=1,
        dpi=96
    )

    values = gws.StyleValues(
        with_geometry='all',
        fill='yellow',
        stroke='black',
        stroke_width=1
    )
    style = gws.Style()
    style.values = values
    
    fragment = draw.shape_to_fragment(shape, view, style=style)
    assert len(fragment) == 1
    assert fragment[0].name == 'path'
    assert 'M 377952.0 3401574.0 L 755905.0 3401574.0 L 755905.0 3023622.0 L 377952.0 3023622.0 L 377952.0 3401574.0 z' in fragment[0].attr('d')
    assert fragment[0].attr('fill') == 'yellow'
    assert fragment[0].attr('stroke') == 'black'
    assert fragment[0].attr('fill-rule') == 'evenodd'


# def test_shape_to_fragment_with_label():
#     """Test shape with label."""
#     shape = gws.base.shape.from_wkt('SRID=4326; POINT(100 200)')
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
#     fragment = draw.shape_to_fragment(shape, view, label="Test Label", style=style)
#     assert len(fragment) == 2  # Point and label
#
#     # Find the label group
#     label_group = None
#     for f in fragment:
#         print(f.__dict__)
#     for el in fragment:
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
    shape = gws.base.shape.from_wkt('SRID=4326; LINESTRING(100 100, 200 200)')
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(
        bounds=bounds,
        size=[1000, 1000],
        rotation=0,
        scale=1,
        dpi=96
    )

    values = gws.StyleValues(
        with_geometry='all',
        marker='circle',
        marker_size=16,
        marker_fill='blue',
        stroke='black'
    )
    style = gws.Style()
    style.values = values
    
    fragment = draw.shape_to_fragment(shape, view, style=style)
    assert len(fragment) == 2  # Marker and path
    
    # Check marker
    marker = fragment[0]
    assert marker.name == 'marker'
    assert marker.attr('markerWidth') == '16'
    assert marker.attr('markerHeight') == '16'
    
    # Check circle inside marker
    circles = [c for c in marker.children() if c.name == 'circle']
    assert len(circles) == 1
    assert circles[0].attr('fill') == 'blue'
    
    # Check path with marker references
    path = fragment[1]
    assert path.name == 'path'
    assert path.attr('marker-start').startswith('url(#_M')
    assert path.attr('marker-mid').startswith('url(#_M')
    assert path.attr('marker-end').startswith('url(#_M')


def test_soup_to_fragment():
    """Test converting a soup to SVG fragment."""
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(
        bounds=bounds,
        size=[1000, 1000],
        rotation=0,
        scale=1,
        dpi=96
    )
    
    points = [(100, 100), (200, 200), (300, 100)]
    tags = [
        ('line', {
            'x1': ['x', 0],
            'y1': ['y', 0],
            'x2': ['x', 1],
            'y2': ['y', 1],
            'stroke': 'black'
        }),
        ('text', {
            'x': ['x', 2],
            'y': ['y', 2],
            'transform': ['r', 0, 1, 2]
        }, 'Test Text')
    ]
    
    fragment = draw.soup_to_fragment(view, points, tags)
    assert len(fragment) == 2
    
    # Check line
    line = fragment[0]
    assert line.name == 'line'
    assert line.attr('x1') == '377952'
    assert line.attr('y1') == '3401574'
    assert line.attr('x2') == '755905'
    assert line.attr('y2') == '3023622'
    assert line.attr('stroke') == 'black'
    
    # Check text with rotation
    text = fragment[1]
    assert text.name == 'text'
    assert text.attr('x') == '1133858'
    assert text.attr('y') == '3401574'
    
    # Check rotation transform
    transform = text.attr('transform')
    assert transform.startswith('rotate(')
    assert '1133858' in transform
    assert '3401574' in transform


def test_empty_shape():
    """Test handling of empty shapes."""
    shape = gws.base.shape.from_wkt('SRID=4326;POLYGON EMPTY')
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(
        bounds=bounds,
        size=[1000, 1000],
        rotation=0,
        scale=1,
        dpi=96
    )
    
    fragment = draw.shape_to_fragment(shape, view)
    assert fragment == []


def test_label_visibility():
    """Test label visibility based on scale."""
    shape = gws.base.shape.from_wkt('SRID=4326;POINT(100 200)')
    
    # Create view with scale 1:1000
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(
        bounds=bounds,
        size=[1000, 1000],
        rotation=0,
        scale=1000,
        dpi=96
    )
    
    # Style with label visible only at scales between 1:500 and 1:2000

    values = gws.StyleValues(
        with_geometry='all',
        with_label='all',
        label_min_scale=500,
        label_max_scale=2000,
        point_size=10,
        fill='red',
        label_font_size=12,
        label_fill='black'
    )
    style = gws.Style()
    style.values = values

    # Label should be visible at scale 1:1000
    fragment = draw.shape_to_fragment(shape, view, label="Test Label", style=style)
    assert len(fragment) == 2  # Point and label
    
    # Create view with scale 1:3000 (outside max_scale)
    view.scale = 3000
    fragment = draw.shape_to_fragment(shape, view, label="Test Label", style=style)
    assert len(fragment) == 1  # Only point, no label
    
    # Create view with scale 1:100 (outside min_scale)
    view.scale = 100
    fragment = draw.shape_to_fragment(shape, view, label="Test Label", style=style)
    assert len(fragment) == 1  # Only point, no label


def test_multigeometry():
    """Test handling of multi-geometries."""
    shape = gws.base.shape.from_wkt('SRID=4326;MULTIPOINT((100 100), (200 200))')
    bounds = gws.Bounds(crs=crs.WGS84, extent=[0, 0, 1000, 1000])
    view = gws.MapView(
        bounds=bounds,
        size=[1000, 1000],
        rotation=0,
        scale=1,
        dpi=96
    )

    values = gws.StyleValues(
        with_geometry='all',
        point_size=10,
        fill='red'
    )
    style = gws.Style()
    style.values = values
    
    fragment = draw.shape_to_fragment(shape, view, style=style)
    assert len(fragment) == 1
    assert fragment[0].name == 'g'
    
    # Check that the group contains two circles
    circles = [c for c in fragment[0].children() if c.name == 'circle']
    assert len(circles) == 2
    
    # Check coordinates of the circles
    coords = [(c.attr('cx'), c.attr('cy')) for c in circles]
    assert coords == [('377952', '3401574'), ('755905', '3023622')]
