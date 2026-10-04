"""WFS client.

Uses the feature types of a remote WFS service as vector layers, search
sources and models. Requests are built for WFS 1.x or WFS 2.0, depending on
the version reported in the capabilities.

Submodules:

- ``provider``: the WFS service provider. Reads the capabilities and runs
  GetFeature requests.
- ``caps``: parser for the WFS capabilities document. Each feature type
  becomes a queryable source layer.
- ``layer``: the ``wfs`` tree layer, a group with a ``wfsflat`` child layer
  for each feature type.
- ``flatlayer``: the ``wfsflat`` vector layer that shows the features of a
  single feature type.
- ``finder``: the ``wfs`` finder that searches feature types by geometry.
- ``model``: the ``wfs`` read-only model that reads features from feature types.

The ``wfsflat`` layer creates a ``wfs`` model and a ``wfs`` finder for its
feature type, unless models or finders are configured explicitly. Unless
configured otherwise, it uses the WGS84 extent of the feature type and, if
there is exactly one source layer, its metadata. The ``wfs`` tree layer uses
the service metadata, unless metadata is configured.

GetFeature requests are spatial only: the provider requests the features
within the bounds of the search shape (enlarged by the search tolerance) with
a ``BBOX`` parameter and then filters them on the GWS side by intersecting
them with the shape. This is faster than WFS spatial operators (at least with
QGIS Server) and works with any WFS server, including those without support
for spatial filters. A search without bounds or shape returns nothing.
Requests are sent in the forced CRS of the provider, or in WGS84. For WFS 2
the CRS is appended to the ``BBOX`` value by default, which can be changed
with ``withBboxCrs``.

References:

- WFS 1.0.0: http://portal.opengeospatial.org/files/?artifact_id=7176 Sec 13.7.3
- WFS 1.1.0: http://portal.opengeospatial.org/files/?artifact_id=8339 Sec 14.7.3
- WFS 2.0.0: http://docs.opengeospatial.org/is/09-025r2/09-025r2.html Sec 11.1.3
- https://docs.geoserver.org/latest/en/user/services/wfs/reference.html

Example::

    map.layers+ {
        title "Districts"
        type "wfsflat"
        provider.url "https://example.com/wfs"
        sourceLayers.names ["ns:districts"]
    }

    map.layers+ {
        title "All feature types"
        type "wfs"
        provider.url "https://example.com/wfs"
    }
"""
