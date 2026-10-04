"""GML geometry support.

Reads GML geometry elements into ``gws.Shape`` objects and ``gws.Bounds``, and writes
shapes as GML elements. Both GML 2 and GML 3 are supported. Only 2D coordinates are handled.

Submodules:

- ``parser``: parses ``gml:Box``/``gml:Envelope`` into bounds and geometry elements
  (``Point``, ``LineString``, ``Polygon``, ``Curve``, ``Multi*`` and so on) into shapes
  or GeoJSON-like geometry dicts.
- ``writer``: converts a shape to a GML 2 or GML 3 geometry element.

Both work on ``gws.XmlElement`` objects from ``gws.lib.xmlx``. The parser converts
GML to a GeoJSON-like dict first and creates the shape with ``gws.lib.shape.from_geojson``.
The CRS is taken from the ``srsName`` attribute of the element, or from a default CRS
passed by the caller. The axis order follows the CRS, unless ``always_xy`` is set.

Reference:
    - https://www.ogc.org/standards/gml

Example::

    el = gws.lib.xmlx.from_string('<Point srsName="EPSG:3857"><pos>1 2</pos></Point>')
    shape = gws.lib.gml.parse_shape(el)
    el = gws.lib.gml.shape_to_element(shape, version=2)
"""

from .parser import parse_envelope, parse_shape, parse_geometry, is_geometry_element
from .writer import shape_to_element
