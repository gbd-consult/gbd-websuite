"""Legend types.

A legend is configured with ``legend`` on a layer. Each legend type renders
one image. Several images are combined vertically into one with
``gws.base.legend.combine_outputs``. Layers from WMS services and QGIS
projects create their legends automatically if no ``legend`` is configured.

Subpackages
-----------

- ``combined`` - combines the legends of other layers, given by uid.
- ``html`` - renders an HTML template to a PNG image.
- ``remote`` - downloads images from URLs and combines them.
- ``static`` - an image file.

Example::

    map.layers+ {
        title "OpenStreetMap"
        type "tile"
        provider.url "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        legend {
            type "static"
            path "osm_legend.png"
        }
    }
"""
