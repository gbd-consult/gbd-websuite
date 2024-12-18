"""GML support.


GML documentation: https://www.ogc.org/standards/gml
"""

from .parser import parse_envelope, parse_shape, parse_geometry, is_geometry_element
from .writer import shape_to_element
