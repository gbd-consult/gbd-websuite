"""Place and address search with Nominatim.

This plugin searches places and addresses in OpenStreetMap through the public
Nominatim service (https://nominatim.org/release-docs/develop/api/Search/).

- ``finder``: the ``nominatim`` finder. It creates a ``nominatim`` model and
  provides default teaser and description templates.
- ``model``: the read-only ``nominatim`` model. It sends the search keyword to
  the service, limited to the bounds of the search shape (in WGS84) and
  optionally to countries, with an optional preferred language, and turns the
  results into features, sorted by name, OSM class and OSM type.

The results are features with WGS84 geometries transformed to the search CRS.
Results that do not intersect the search shape are dropped. The address parts
of a result are flattened to ``address_*`` attributes, and the attributes
``name``, ``osm_class`` and ``osm_type`` are added for use in templates.

Example::

    actions+ { type "search" }

    finders+ {
        type "nominatim"
        spatialContext "map"
        country "de"
        language "de"
    }
"""
