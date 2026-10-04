"""Feature exporters.

Exporters write features to files in vector formats, with GDAL vector drivers
(``gws.base.exporter.util.run_gdal_vector_export``). Features are grouped by
model. Each group is written to its own file or, with ``withMultiLayer`` and a
format that supports it, to its own layer in a single file. Several files are
zipped. ``options`` are passed to the GDAL driver as creation options.

Subpackages
-----------

- ``csv`` - CSV.
- ``geojson`` - GeoJSON.
- ``gml`` - GML, supports several layers in one file.
- ``kml`` - KML, supports several layers in one file.
- ``shapefile`` - ESRI Shapefile.

Example::

    exporters+ {
        type "geojson"
        title "GeoJSON"
        target "download"
    }

    exporters+ {
        type "gml"
        title "GML, all layers in one file"
        target "download"
        withMultiLayer true
    }
"""
