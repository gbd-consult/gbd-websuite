"""OGC web services provided by GBD WebSuite.

This plugin publishes projects and their layers as OGC services. The common
parts (request parsing, dispatching, layer caps, error responses, template
helpers and the ``ows`` action) live in ``gws.base.ows.server``; the subpackages
here implement the protocols.

Subpackages:

- ``wms``: WMS 1.1.x and 1.3.0 service. Renders map images (GetMap), returns
  feature info (GetFeatureInfo) and legends (GetLegendGraphic).
- ``wmts``: WMTS 1.0.0 service. Renders the layers as tiles on a tile matrix
  set for each supported CRS.
- ``wfs``: WFS 2.0 service. Returns features of searchable layers as GML or
  GeoJSON.
- ``csw``: CSW 2.0.2 catalogue service. Publishes the metadata of public
  objects that have a catalog UID as ISO 19115 records.

Each service is an ``owsService`` object. It declares its supported operations,
each with the name of a handler method, and the base class dispatches incoming
requests to these handlers. XML responses (capabilities, feature collections,
records) are produced by templates of type ``py``; each service package has a
``templates`` subpackage with the default templates, which can be replaced by
configuring templates with the same subjects.

Example::

    actions+ { type "ows" }

    projects+ {
        uid "my_project"
        owsServices+ { type "wms" uid "my_wms" }
        owsServices+ { type "wmts" uid "my_wmts" }
        owsServices+ { type "wfs" uid "my_wfs" }
    }

    owsServices+ { type "csw" uid "my_csw" }
"""
