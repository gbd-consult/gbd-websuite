QGIS support
============

GBD WebSuite offers dedicated support for `QGIS <https://qgis.org>`_, a free and open source geographic information system. The support is optional, and can be turned off if you don't use QGIS.

Our QGIS module allows you to

- display complete QGIS projects (``.qgs``) as *layers* in your GWS project (see :doc:`layers`)
- use QGIS projects (``.qgs``) as *sources* for your image layers (see :doc:`sources`)
- use QGIS templates for printing (see :doc:`print`)

In the server configuation (:doc:`server`) there are a few options that set the values of QGIS environment variables. Refer to the QGIS documentation for their specific meaning:

TABLE
*debug*	~ QGIS_DEBUG
*maxCacheLayers* ~ MAX_CACHE_LAYERS
*serverCacheSize* ~ QGIS_SERVER_CACHE_SIZE
*serverLogLevel* ~ QGIS_SERVER_LOG_LEVEL
/TABLE

The ``searchPathsForSVG`` option tells where to find svg images used in your QGIS projects and print templates. If you use non-standard images, just add a directory path for them to this setting.
 
