"""GBD Geoservices search.

Keyword and location search with the GBD Geoservices search service
(https://geoservices.gbd-consult.de), which finds places, streets, addresses
and points of interest. Requests are authenticated with an API key.

Submodules
----------

- ``finder`` - the ``gbd_geoservices`` finder. It supports keyword and
  geometry search, has default templates for the feature title, teaser and
  description, and creates a ``gbd_geoservices`` model with its API key.
- ``model`` - the ``gbd_geoservices`` model. It sends the search to the
  service and converts the results into read-only features.
- ``templates`` - the default feature templates.
- ``js`` - the client part.

A search with a keyword is sent as a phrase search; the search shape, if any,
is sent along as the search area. A search without a keyword, but with a
shape, is sent as a point search, with the tolerance as the radius (at most
10 km). At most 100 results are returned. The features get the attributes
``title``, ``subtitle``, ``teaser``, ``details`` and ``icon``, built from the
name, category, address and places of a result.

Example::

    finders+ {
        type "gbd_geoservices"
        apiKey "..."
        spatialContext "map"
    }
"""
