"""GIS functionality built on the libraries.

This package holds GIS code that is more specific than the general purpose
libraries in ``gws.lib`` and is meant to be independent of the configurable
objects in ``gws.base``. It sits between ``lib`` and ``base`` in the import
hierarchy.

Subpackages:

- ``cache``: tile cache management: the cache configuration, the filesystem
  tile store used by grabbers, and the ``gws cache`` commands to inspect,
  seed and drop caches.
- ``render``: map rendering for printing, exporting and templates. Renders
  input planes (image layers, SVG layers, features, images) into alternating
  raster and SVG output planes, and converts the output to HTML.
- ``source``: source layers of external services (WMS, WFS, WMTS, QGIS), read
  from their capabilities, and the filters that select them.
- ``zoom``: zoom levels of maps and layers: resolutions computed from the
  ``zoom`` configuration and conversion between scales and resolutions.

How they relate:

``zoom`` provides the resolutions that maps and layers are configured with.
``source`` describes what an external service offers; layers in ``gws.base``
and in plugins pick their source layers from it. ``render`` renders a map
view by asking each layer for a box image or SVG elements; image layers
produce their images through grabbers, which keep their tiles in the
``cache`` stores.

Example::

    import gws
    import gws.gis.render
    import gws.gis.zoom
    import gws.lib.crs

    scale = gws.gis.zoom.res_to_scale(10, gws.lib.crs.WEBMERCATOR)

    view = gws.gis.render.map_view_from_center(
        size=(200, 150, gws.Uom.mm),
        center=(1000, 2000),
        crs=gws.lib.crs.WEBMERCATOR,
        dpi=150,
        scale=10000,
    )
"""
