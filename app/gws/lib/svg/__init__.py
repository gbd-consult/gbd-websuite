"""SVG drawing and sanitizing.

This package creates SVG output for vector features, used when rendering maps and prints.
The basic unit is a *fragment*: a list of ``gws.XmlElement`` objects that are not yet wrapped
in an ``<svg>`` root element. Fragments from several features can be joined and then converted
to a single SVG element or a raster image.

Submodules:

- ``draw``: builds fragments. ``shape_to_fragment`` draws a shape with a ``gws.Style``
  (geometry, markers, icons and labels), transformed to pixels by a ``gws.MapView``.
  ``soup_to_fragment`` builds a fragment from a "soup", a resolution-independent description
  of client-side drawings (e.g. dimensions).
- ``element``: wraps fragments into SVG elements or images (``fragment_to_element``,
  ``fragment_to_image``) and removes unsafe content from SVG input (``normalize_element``,
  ``normalize_fragment``), keeping only allowed tags and attributes with valid values.

Labels are emitted with a ``z-index`` attribute, so that ``fragment_to_element`` can place them above the geometries.

Example::

    import gws.lib.svg

    frag = []
    for feature in features:
        frag.extend(gws.lib.svg.shape_to_fragment(feature.shape(), view, label='A', style=style))
    el = gws.lib.svg.fragment_to_element(frag)
    img = gws.lib.svg.fragment_to_image(frag, (800, 600))
"""

from .draw import shape_to_fragment, soup_to_fragment
from .element import fragment_to_element, fragment_to_image, normalize_element, normalize_fragment
