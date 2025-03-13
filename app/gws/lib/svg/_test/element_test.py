"""Tests for the SVG element module."""

import io
import pytest

import gws
import gws.lib.xmlx as xmlx
import gws.lib.image
import gws.lib.mime
import gws.lib.svg.element as svg_element


def test_fragment_to_element_basic():
    """Test basic fragment to element conversion."""
    # Create a simple fragment                                                                                                                                                         
    circle = xmlx.tag('circle', {'cx': 50, 'cy': 50, 'r': 25, 'fill': 'red'})
    rect = xmlx.tag('rect', {'x': 10, 'y': 10, 'width': 80, 'height': 80, 'fill': 'blue'})

    # Convert to SVG element                                                                                                                                                           
    svg = svg_element.fragment_to_element([circle, rect])

    # Verify the result                                                                                                                                                                
    assert svg.name == 'svg'
    assert svg.attr('xmlns') == 'http://www.w3.org/2000/svg'
    assert len(svg.children()) == 2
    assert svg.children()[0].name == 'circle'
    assert svg.children()[1].name == 'rect'


def test_fragment_to_element_with_attributes():
    """Test fragment to element conversion with custom attributes."""
    circle = xmlx.tag('circle', {'cx': 50, 'cy': 50, 'r': 25})

    # Add custom attributes                                                                                                                                                            
    custom_atts = {'width': '100', 'height': '100', 'viewBox': '0 0 100 100'}
    svg = svg_element.fragment_to_element([circle], custom_atts)

    # Verify attributes were applied                                                                                                                                                   
    assert svg.attr('width') == '100'
    assert svg.attr('height') == '100'
    assert svg.attr('viewBox') == '0 0 100 100'


def test_fragment_to_element_z_index_sorting():
    """Test that elements are sorted by z-index."""
    # Create elements with different z-indices                                                                                                                                         
    back = xmlx.tag('rect', {'x': 0, 'y': 0, 'width': 100, 'height': 100, 'fill': 'blue', 'z-index': 1})
    middle = xmlx.tag('circle', {'cx': 50, 'cy': 50, 'r': 40, 'fill': 'green', 'z-index': 2})
    front = xmlx.tag('circle', {'cx': 50, 'cy': 50, 'r': 20, 'fill': 'red', 'z-index': 3})

    # Add in reverse order                                                                                                                                                             
    svg = svg_element.fragment_to_element([front, back, middle])

    # Verify they're sorted by z-index                                                                                                                                                 
    children = svg.children()
    assert children[0].attrib.get(
        'fill') == 'blue'  # z-index: 1
    assert children[1].attrib.get(
        'fill') == 'green'  # z-index: 2
    assert children[2].attrib.get(
        'fill') == 'red'  # z-index: 3


def test_fragment_to_element_empty_fragment():
    """Test conversion with an empty fragment."""
    svg = svg_element.fragment_to_element([])

    # Should still create an SVG element, just without children                                                                                                                        
    assert svg.name == 'svg'
    assert len(svg.children()) == 0


# def test_fragment_to_image():
#   Returnfunction is not implemented yet
#   gws.lib.image.from_svg(el.to_string(), size, mime)


def test_sanitize_element_allowed_tags():
    """Test sanitization of allowed tags."""
    # Create an SVG with allowed tags                                                                                                                                                  
    svg = xmlx.tag('svg',
                   {'width': '100', 'height': '100'},
                   xmlx.tag('circle', {'cx': '50', 'cy': '50', 'r': '40', 'fill': 'blue'}),
                   xmlx.tag('rect', {'x': '10', 'y': '10', 'width': '80', 'height': '80', 'fill': 'red'})
                   )

    # Sanitize                                                                                                                                                                         
    result = svg_element.sanitize_element(svg)

    # Verify allowed tags are preserved                                                                                                                                                
    assert result is not None
    assert len(result.children()) == 2
    assert result.children()[0].name == 'circle'
    assert result.children()[1].name == 'rect'


def test_sanitize_element_disallowed_tags():
    """Test sanitization of disallowed tags."""
    # Create an SVG with a disallowed tag (script)                                                                                                                                     
    svg = xmlx.tag('svg',
                   {'width': '100', 'height': '100'},
                   xmlx.tag('circle', {'cx': '50', 'cy': '50', 'r': '40'}),
                   xmlx.tag('script', {}, "alert('XSS attack');")
                   )

    # Sanitize                                                                                                                                                                         
    result = svg_element.sanitize_element(svg)

    # Verify disallowed tags are removed                                                                                                                                               
    assert result is not None
    assert len(result.children()) == 1
    assert result.children()[0].name == 'circle'


def test_sanitize_element_allowed_attributes():
    """Test sanitization of allowed attributes."""
    # Create an element with allowed attributes                                                                                                                                        
    svg = xmlx.tag('svg', {'width': '100', 'height': '100'},
                   xmlx.tag('circle', {
                       'cx': '50',
                       'cy': '50',
                       'r': '40',
                       'fill': 'blue',
                       'stroke': 'black',
                       'stroke-width': '2'
                   })
                   )

    # Sanitize                                                                                                                                                                         
    result = svg_element.sanitize_element(svg)

    # Verify allowed attributes are preserved                                                                                                                                          
    circle = result.children()[0]
    assert circle.attr('fill') == 'blue'
    assert circle.attr('stroke') == 'black'
    assert circle.attr('stroke-width') == '2'


def test_sanitize_element_disallowed_attributes():
    """Test sanitization of disallowed attributes."""
    # Create an element with disallowed attributes                                                                                                                                     
    svg = xmlx.tag('svg', {'width': '100', 'height': '100'},
                   xmlx.tag('circle', {
                       'cx': '50',
                       'cy': '50',
                       'r': '40',
                       'fill': 'blue',
                       'onmouseover': "alert('XSS attack');",
                       # Disallowed
                       'onclick': "maliciousFunction();"
                       # Disallowed
                   })
                   )

    # Sanitize                                                                                                                                                                         
    result = svg_element.sanitize_element(svg)

    # Verify disallowed attributes are removed                                                                                                                                         
    circle = result.children()[0]
    assert circle.attr('fill') == 'blue'
    assert circle.attr('onmouseover') is None
    assert circle.attr('onclick') is None


def test_sanitize_element_url_attributes():
    """Test sanitization of URL attributes."""
    # Create an element with URL attributes                                                                                                                                            
    svg = xmlx.tag('svg', {'width': '100', 'height': '100'},
                   xmlx.tag('circle', {
                       'cx': '50',
                       'cy': '50',
                       'r': '40',
                       'fill': 'url(http://evil.com/script.js)',
                       # Should be removed
                       'stroke': 'black'
                   })
                   )

    # Sanitize                                                                                                                                                                         
    result = svg_element.sanitize_element(svg)

    # Verify URL attributes are removed                                                                                                                                                
    circle = result.children()[0]
    assert circle.attr('fill') is None
    assert circle.attr('stroke') == 'black'


def test_sanitize_element_nested_structure():
    """Test sanitization of nested elements."""
    # Create a nested structure with allowed and disallowed elements                                                                                                                   
    svg = xmlx.tag('svg', {'width': '100', 'height': '100'},
                   xmlx.tag('g', {'transform': 'translate(10,10)'},
                            xmlx.tag('circle', {'cx': '50', 'cy': '50', 'r': '40', 'fill': 'blue'}),
                            xmlx.tag('script', {}, "alert('XSS attack');")
                            # Disallowed
                            ),
                   xmlx.tag('foreignObject', {}, "Some foreign content")
                   # Disallowed
                   )

    # Sanitize                                                                                                                                                                         
    result = svg_element.sanitize_element(svg)

    # Verify structure is preserved but disallowed elements are removed                                                                                                                
    assert result is not None
    assert len(result.children()) == 1

    g = result.children()[0]
    assert g.name == 'g'
    assert len(g.children()) == 1
    assert g.children()[0].name == 'circle'


def test_sanitize_element_empty_result():
    """Test sanitization that results in an empty SVG."""
    # Create an SVG with only disallowed tags                                                                                                                                          
    svg = xmlx.tag('svg', {'width': '100', 'height': '100'},
                   xmlx.tag('script', {}, "alert('XSS attack');"),
                   xmlx.tag('foreignObject', {}, "Some foreign content")
                   )

    # Sanitize                                                                                                                                                                         
    result = svg_element.sanitize_element(svg)

    # Verify we get an empty SVG                                                                                                                                                       
    assert not result


def test_sanitize_element_data_url():
    """Test sanitization of data URLs."""
    # Create an element with a data URL                                                                                                                                                
    svg = xmlx.tag('svg', {'width': '100', 'height': '100'},
                   xmlx.tag('circle', {
                       'cx': '50',
                       'cy': '50',
                       'r': '40',
                       'fill': 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxzY3JpcHQ+YWxlcnQoMSk8L3NjcmlwdD48L3N2Zz4='
                   })
                   )

    # Sanitize                                                                                                                                                                         
    result = svg_element.sanitize_element(svg)

    # Verify data URL is removed                                                                                                                                                       
    circle = result.children()[0]
    assert circle.attr('fill') is None 