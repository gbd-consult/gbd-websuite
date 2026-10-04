"""Maps and the map action.

A map holds the root of a layer tree together with the CRS, extent, center and
the resolutions (zoom levels) shared by its layers. Each project has a map.
The map's layers are created as children of an implicit ``group`` root layer,
which passes the map extent, resolutions and CRS down the tree (see
``gws.base.layer``).

Submodules:

- ``core``: the ``default`` map object with its ``Config`` and ``Props``.
  The CRS defaults to Web Mercator. Without an ``extent`` the map covers the
  maximum extent of its CRS; the center defaults to the center of the extent.
  The title defaults to the title of the project.
- ``action``: the ``map`` action. Its commands serve layer images as boxes
  (``mapGetBox``) and tiles (``mapGetTile``), legends (``mapGetLegend``),
  layer descriptions (``mapDescribeLayer``) and features (``mapGetFeatures``)
  to the client. Layers build the URLs for these commands with
  ``Layer.url_path_for``. Image commands answer with a transparent pixel
  when the layer renders nothing or rendering fails; the failure is logged.

Example::

    actions+ { type "map" }

    projects+ {
        title "City map"
        map {
            crs 25832
            extent [ 280000 5640000 300000 5660000 ]
            zoom.scales [ 100000 50000 25000 10000 5000 ]
            zoom.initScale 25000
            layers+ { type "qgis" provider.path "/data/city.qgs" }
        }
    }
"""

from .core import Config, Object, Props
