"""Search.

Searching finds features by a keyword, a geometry or a filter. A search is
described by a ``gws.SearchQuery`` and run by finders (``gws.Finder``), which are
configured for layers, projects and the application.

Submodules:

- ``finder``: the base class for finders, with the common ``Config`` options
  (``withKeyword``, ``withGeometry``, ``withFilter``, ``spatialContext`` and others).
  Concrete finders live in plugins (for example ``postgres``, ``qgis``, ``wms``,
  ``wfs``, ``nominatim``).
- ``manager``: the search manager (``gws.SearchManager``), which runs a query
  through all finders that apply.
- ``action``: the ``search`` action, whose ``searchFind`` command runs a search
  for the client and returns the found features with their rendered views.
- ``filter``: a parser for OGC Filter Encoding 2.0 (FES) filters into
  ``gws.SearchFilter`` objects, and a ``Matcher`` that evaluates such a filter
  against Python objects.

The manager runs the finders of the requested layers first, then the finders of
the project, then the application finders, and stops when the query limit is
reached. A finder is skipped if the user may not use it, or if it cannot handle
the query: every search parameter given in the query (keyword, shape, filter)
must be supported by the finder and enabled in its config. A finder that fails
is logged and skipped. Each found feature gets a category: the finder category,
the finder title, or, if the feature has none, the layer title.

A finder reads features through a model it finds for itself and the layer.
Keyword searches without a user geometry are restricted to the area given by
``spatialContext``: the whole map or the current client view.

The ``filter`` module supports the minimum standard filter (``PropertyIsEqualTo``,
``PropertyIsNotEqualTo``, ``PropertyIsLessThan``, ``PropertyIsGreaterThan``,
``PropertyIsLessThanOrEqualTo``, ``PropertyIsGreaterThanOrEqualTo``) with the
logical operators ``And``, ``Or`` and ``Not``, and, of the spatial operators,
only ``BBOX``. Property names can be plain or namespace-prefixed. See the
`OGC Filter Encoding 2.0 standard <http://docs.opengeospatial.org/is/09-026r2/09-026r2.html>`_.

Example::

    {
        actions [
            { type "search" limit 100 tolerance "10px" }
        ]
        projects [
            {
                uid "project_1"
                finders [
                    { type "postgres" tableName "public.streets" withGeometry false }
                ]
            }
        ]
    }

Example::

    search = gws.SearchQuery(project=project, keyword='Main street', limit=10)
    results = root.app.searchMgr.run_search(search, user)
    for res in results:
        print(res.feature.uid, res.finder.uid)

Example::

    flt = gws.base.search.filter.from_fes_string(xml_string)
    matcher = gws.base.search.filter.Matcher()
    found = [obj for obj in objects if matcher.matches(flt, obj)]
"""

from . import manager, finder
