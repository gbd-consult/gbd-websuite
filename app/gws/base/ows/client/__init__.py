"""Base classes and utilities for OWS clients.

This package provides the protocol-independent parts of the OWS clients. The
protocol plugins (``gws.plugin.ows_client.wms``, ``wmts``, ``wfs``) and the
QGIS plugin subclass the provider, finder and model defined here and use the
parsing helpers to read capabilities documents.

Submodules:

- ``provider``: base ``OwsServiceProvider``. Holds the service URL, version,
  operations and source layers, merges the operations reported in the
  capabilities with those in the configuration, selects a preferred format for
  each operation, prepares request arguments (including basic authorization)
  and downloads and caches the capabilities document.
- ``request``: low-level HTTP helpers. Build the ``SERVICE``, ``REQUEST`` and
  ``VERSION`` parameters, send the request and turn OWS exception documents
  into ``gws.ExternalServiceError``, also when the server answers with status 200.
- ``parseutil``: helpers to read capabilities XML: operations, service and
  layer metadata, WGS84 extents, supported CRSs, styles and number conversion.
- ``featureinfo``: parses GetFeatureInfo and GetFeature responses in several
  XML formats (GML feature collections, MapServer ``msGMLOutput``,
  GeoServer/QGIS ``GetFeatureInfoResponse``, ArcGIS ``FeatureInfoResponse``,
  GeoBAK, OSIRIS) into ``gws.FeatureRecord`` objects.
- ``finder``: generic finder that searches the queryable source layers of a provider.
- ``model``: generic read-only model that reads features from a provider.
- ``cli``: the ``owsCaps`` CLI command that prints the parsed capabilities of a
  service or a local XML file as JSON. It is not imported by this package.

A provider subclass typically downloads the capabilities in ``configure``,
parses them with a protocol-specific parser built on ``parseutil``, calls
``configure_operations`` with the parsed operations, and later uses
``get_operation`` and ``prepare_operation`` to build requests, which are sent
with ``request.get``.

Example::

    op = provider.get_operation(gws.OwsVerb.GetFeatureInfo)
    args = provider.prepare_operation(op, params={'QUERY_LAYERS': 'roads'})
    text = gws.base.ows.client.request.get_text(args)
    records = gws.base.ows.client.featureinfo.parse(text, default_crs=provider.forceCrs)
"""

from . import (
    featureinfo,
    finder,
    model,
    parseutil,
    provider,
    request,
)
