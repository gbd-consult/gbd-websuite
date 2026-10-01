"""Map share tool.

The client tool ``Toolbar.MapShare`` creates a link to a clicked map position.
Links are created and decoded by the ``mapshare`` action, in the form ``/project/<uid>?s=<link>``, where ``<link>`` is
``<scale>x<x>y<y>`` for points or ``<scale>w<wkb hex>`` for other geometries, optionally followed by ``_<title>``.
Coordinates are in the project map CRS, rounded to meters (5 decimals for geographic CRS).
"""
