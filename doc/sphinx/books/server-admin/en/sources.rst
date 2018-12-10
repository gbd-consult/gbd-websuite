Sources
=======

A *source* describes where the geo-data comes from. There are different types of sources. Note that not all sources can be used with all layer types. For instance, you cannot use a WMS source with a Vector layer.

Source types
------------

wms
~~~

A WMS source  provides WMS imagery from an external service. In addition to the service url and parameters, you can also configure ``maxRequests`` to prevent GWS from scraping the service excessively, especially when seeding the caches


wmts
~~~~

A WMTS source is similar to WMS, but works with WMTS services.

tile
~~~~

Tile sources work with XYZ tile services.

QGIS
~~~~

QGIS sources use QGIS projects (``.qgs`` files) as data and imagery sources. You can use mutiple QGIS sources in your project and freely combine them with other source types. The ``layers`` option tells which layers from ``.qgs`` to include in the project, set ``allLayers`` to ``true`` to include all layers.

Note: QGIS maintains layer names and ids as separate fields. You can use either for the ``layers`` list, but when using ids, make sure the QGIS option "Use layer ids as names" is on.


geoJSON
~~~~~~~

A geoJSON source is a file that can be used with Vector layers.

sql
~~~

A SQL source describes a DB provider and table to get the data from. This source can be used with Vector layers.
