"""OGC Web Services (OWS).

Common base for the OWS support in GBD WebSuite: the client side, which reads data
from external OWS services, and the server side, which publishes project layers
as OWS services. The protocols themselves (WMS, WMTS, WFS, CSW) are implemented
as plugins in ``gws.plugin.ows_client`` and ``gws.plugin.ows_server``, built on
the classes in this package.

Subpackages:

- ``client``: base classes and helpers for OWS clients: the service provider
  (``provider``), the HTTP request helpers (``request``), capabilities parsing
  utilities (``parseutil``), the FeatureInfo response parser (``featureinfo``),
  a generic finder (``finder``) and model (``model``) backed by a provider,
  and the ``owsCaps`` CLI command (``cli``).
- ``server``: base classes and helpers for OWS services: the service base class
  (``service``), the service request (``request``), layer capabilities
  (``layer_caps``, ``core``), OWS exceptions (``error``), helpers for service
  templates (``templatelib``) and the ``ows`` web action (``action``).

The interfaces shared by both sides (``gws.OwsProtocol``, ``gws.OwsVerb``,
``gws.OwsOperation``, ``gws.OwsService``, ``gws.OwsServiceProvider`` and others)
are defined in ``types.pyinc`` and included into the ``gws`` module.

Example::

    import gws.base.ows.client
    import gws.base.ows.server

    # client: send a raw OWS request
    args = gws.base.ows.client.request.Args(
        url='https://example.com/wms',
        protocol=gws.OwsProtocol.WMS,
        verb=gws.OwsVerb.GetCapabilities,
    )
    xml = gws.base.ows.client.request.get_text(args)
"""
