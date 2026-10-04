"""WMS client.

Uses the layers of a remote WMS service (versions 1.1.x and 1.3.0) as image
layers, search sources and models.

Submodules:

- ``provider``: the WMS service provider. Reads the capabilities and runs
  GetMap and GetFeatureInfo requests.
- ``caps``: parser for the WMS capabilities document. Builds the source layer
  tree, including the properties that child layers inherit from their parents.
- ``layer``: the ``wms`` tree layer, a group that mirrors the layer tree of
  the service, with ``wmsflat`` layers as leaves.
- ``flatlayer``: the ``wmsflat`` image layer that renders selected source
  layers as a single image.
- ``grabber``: the box grabber used by ``wmsflat`` to fetch map images with GetMap.
- ``finder``: the ``wms`` finder that queries source layers with
  GetFeatureInfo at a clicked point.
- ``model``: the ``wms`` read-only model for features returned by GetFeatureInfo.

The ``wmsflat`` layer creates a ``wms`` model and, if any of its source layers
is queryable, a ``wms`` finder, unless models or finders are configured
explicitly. Unless configured otherwise, it takes its extent from the image
source layers, its resolutions from the scale ranges of the source layers,
its legend from their legend URLs and, if there is exactly one source layer,
its metadata. The ``wms`` tree layer uses the service metadata, unless
metadata is configured.

Map images are fetched through a grabber, which can cache them. The grabber
requests images in its image format; the size of a single GetMap request is
limited by ``maxRequestPixels`` of the provider.

GetFeatureInfo searches only run for a point shape; other searches return
nothing. The provider requests a 500x500 pixel image centered at the point and
queries its center pixel. The box covers 500 pixels at the search resolution
for metric CRSs and 1 degree for geographic CRSs. The request CRS is the forced
CRS of the provider, or the source CRS that best matches the CRS of the shape.

Layer order: internally, source layers are always listed topmost layer first,
which corresponds to the layer tree display. WMS capabilities are assumed to
be top-first as well. For servers with bottom-first capabilities, set
``bottomFirst`` on the provider; the capabilities parser then reverses all
layer lists. GetMap draws the leftmost layer in the list bottommost (OGC 06-042,
7.3.3.3), so the provider reverses the layer list when it calls GetMap.

References:

- OGC 01-068r3: WMS 1.1.1
- OGC 06-042: WMS 1.3.0
- https://docs.geoserver.org/latest/en/user/services/wms/reference.html

Example::

    map.layers+ {
        title "Districts"
        type "wmsflat"
        provider.url "https://example.com/wms"
        sourceLayers.names ["districts" "roads"]
    }

    map.layers+ {
        title "Service tree"
        type "wms"
        provider.url "https://example.com/wms"
        provider.bottomFirst true
    }
"""
