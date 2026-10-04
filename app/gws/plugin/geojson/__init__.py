"""GeoJSON support.

Layers, models and finders for features stored in a GeoJSON file. The file is
read when it is first needed and kept in memory. Its CRS is taken from the
``crs`` member (a named CRS, as in the 2008 GeoJSON specification); without
it, WGS84 is assumed. The uid of a feature is taken from its ``id``, ``uid``,
``fid`` or ``sid`` property, or else is its position in the file.

Submodules
----------

- ``provider`` - reads the GeoJSON file and selects the records that match a
  search by shape or bounds, keyword and uids.
- ``layer`` - the ``geojson`` vector layer. The geometry type and CRS of the
  layer are taken from the first feature with a geometry, the extent from the
  bounds of all features. By default, the layer has a ``geojson`` model and a
  ``geojson`` finder with the same provider.
- ``model`` - the ``geojson`` model, which returns the features that match a
  search.
- ``finder`` - the ``geojson`` finder, which supports geometry search.

Example::

    map.layers+ {
        type "geojson"
        title "Banks"
        provider.path "/data/banks.geojson"
    }
"""
