"""Map share tool.

The client tool ``Toolbar.MapShare`` creates a link to a clicked map position,
with the current scale and an optional title, and shows it together with a QR
code. Opening the link shows the shared position on the map.

Links have the form ``/project/<uid>?s=<link>``, where ``<link>`` is
``<scale>x<x>y<y>`` for points or ``<scale>w<wkb hex>`` for other geometries,
optionally followed by ``_<title>``. Coordinates are in the CRS of the project
map, rounded to meters (5 decimals for geographic CRS). Titles are cut to 200
characters.

Submodules
----------

- ``action`` - the ``mapshare`` action. Implements ``mapshareCreateLink``,
  which creates the link and its QR code, and ``mapshareDecodeLink``, which
  decodes the ``s`` parameter of a link.
- ``js`` - the client part.

Example::

    actions+ { type "mapshare" }

    client.addElements+ { tag "Toolbar.MapShare" }
"""
