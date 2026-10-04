"""Clients for remote OGC web services.

This plugin lets GBD WebSuite use external WMS, WMTS and WFS services as
data sources for layers, search and models. The protocol-independent parts
(the base provider, request helpers, capabilities parsing utilities, the
generic finder and model) live in ``gws.base.ows.client``; the subpackages
here implement the protocols on top of them.

Subpackages:

- ``wms``: WMS client. Provides the ``wms`` tree layer, the ``wmsflat`` image
  layer, a GetFeatureInfo finder and a model.
- ``wmts``: WMTS client. Provides the ``wmts`` tile layer.
- ``wfs``: WFS client. Provides the ``wfs`` tree layer, the ``wfsflat`` vector
  layer, a GetFeature finder and a model.

Each subpackage has a ``provider`` module with the service provider object.
The provider downloads and parses the capabilities document and performs the
actual requests (GetMap, GetTile, GetFeatureInfo, GetFeature). Layers, finders
and models refer to a provider through their ``provider`` configuration; a
provider created from the same configuration is shared between objects.

Tree layers (``wms``, ``wfs``) create a group with a child layer for each
source layer of the service. The children are flat layers (``wmsflat``,
``wfsflat``) that reuse the provider of the group.

Example::

    map.layers+ {
        title "Base map"
        type "wmts"
        provider.url "https://example.com/wmts/1.0.0/WMTSCapabilities.xml"
        sourceLayers.names ["base_map"]
    }

    map.layers+ {
        title "Districts"
        type "wmsflat"
        provider.url "https://example.com/wms"
        sourceLayers.names ["districts"]
    }

    map.layers+ {
        title "Districts (vector)"
        type "wfsflat"
        provider.url "https://example.com/wfs"
        sourceLayers.names ["ns:districts"]
    }
"""
