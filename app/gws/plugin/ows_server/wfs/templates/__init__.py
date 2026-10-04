"""Default templates of the WFS service.

The templates are ``py`` templates. Each module defines a ``main`` function
that takes the template arguments (``gws.base.ows.server.TemplateArgs``) and
returns the response, mostly built with ``gws.base.ows.server.templatelib``.

Templates:

- ``getCapabilities.cx.py``: the capabilities document (``ows.GetCapabilities``).
- ``getFeature3.cx.py``: feature collection in GML 3 (``ows.GetFeature``).
- ``getFeature2.cx.py``: feature collection in GML 2 (``ows.GetFeature``). Also
  used by the WMS service for GetFeatureInfo.
- ``getFeatureGeoJson.cx.py``: feature collection in GeoJSON (``ows.GetFeature``).
- ``getPropertyValue.cx.py``: property values (``ows.GetPropertyValue``).
- ``listStoredQueries.cx.py``: list of stored queries (``ows.ListStoredQueries``).
- ``describeStoredQueries.cx.py``: description of stored queries (``ows.DescribeStoredQueries``).
"""
