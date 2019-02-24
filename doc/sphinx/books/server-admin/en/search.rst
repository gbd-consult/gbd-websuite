Search
======

GBD WebSuite implements unifed search functionality that works with different types of spatial and domain data. Basically, a search request to the server contains these three parameters:

TABLE
    *keyword* ~ keyword to search for
    *shape* ~ a geometry, if provided, where searches are spatially restricted to
    *layers* ~ a list of layers where to perform the search
/TABLE

When the server receives a search request, it connects to configured data sources and automatically uses the method (or *provider*) applicable to each source. For example, for WMS sources, it issues "GetFeatureInfo" requests. For database data, a "SELECT" query will be peformed and so on. Once the server receives results from different sources, they are consolidated, optionally, reformatted (see :doc:`features`) and returned to the client as a list of uniform GeoJSON features.

Search providers
----------------

nominatim
~~~~~~~~~

Interface to `Nominatim <https://nominatim.openstreetmap.org//>`_, the OSM search engine. You can configure ``country`` and ``language`` parameters to customize search results.

sql
~~~

Provides direct search in SQL (PostGIS) tables. You need to specify the DB provider to use (see :doc:`db`) and the table to search. The table configuration is the table name (optionally, with a schema) and at least one of the following two columns:

- ``searchColumn`` is where to search for the ``keyword``. If not configured, the keyword will be ignored
- ``geometryColumn`` is used to spatially restrict the search. If not configured, the ``shape`` parameter will be ignored


wms
~~~

Implements ``GetFeatureInfo`` requests for WMS sources. You need to provide the WMS service url. It's also possible to restrict the search to specific layers.

wfs
~~~

Implements ``GetFeatureInfo`` requests for WFS sources. You need to provide the WFS service url. It's also possible to restrict the search to specific layers (or "types").

