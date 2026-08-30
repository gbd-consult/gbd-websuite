"""Mockmap: a map server that serves predictable synthetic geometry.

Mockmap exists to test OWS and tile clients. It serves the same synthetic scene
through several protocols and projections, so a client can be checked against
content whose true position is known exactly, without depending on a remote
service or on real imagery.

Running
-------

Standalone, with its own compose file::

    docker compose -f docker-compose.yml up

or inside a GWS container::

    PYTHONPATH=/gws-app python3 -m gws.test.mockmap.server [config path]

Environment: ``MOCKMAP_PORT`` (default 80), ``MOCKMAP_DELAY`` (milliseconds
added to every response, default 0).

Services
--------

One service is one protocol in one CRS with at most one tile grid. Capabilities
are deliberately minimal: mockmap tests rendering, not capabilities parsing.

    ``xyz``         ``/<uid>/{z}/{x}/{y}.png``, north-west origin, no capabilities
    ``tms``         ``/<uid>/1.0.0/<layer>/{z}/{x}/{y}.png``, south-west origin,
                    service and tilemap documents at ``/<uid>/1.0.0[/<layer>]``
    ``wmts``        WMTS KVP: ``GetCapabilities`` and ``GetTile`` on ``/<uid>``
    ``wmts_rest``   WMTS RESTful: ``/<uid>/WMTSCapabilities.xml`` and
                    ``/<uid>/<layer>/<style>/<tms>/{z}/{y}/{x}.png``
    ``wms``         ``GetCapabilities`` and ``GetMap`` on ``/<uid>``, version
                    1.1.1 or 1.3.0

Every request is ``/<service-uid>/`` followed by the service's own path and/or
query parameters::

    /xyz_3857/7/68/44.png
    /wms_32?SERVICE=WMS&REQUEST=GetMap&BBOX=...

``/`` lists the configured service uids. The TileMatrixSet identifier is derived
from the CRS as ``EPSG_<srid>``.

WMS 1.3.0 swaps ``BBOX`` and the advertised ``BoundingBox`` for a CRS with
latitude/northing first; WMTS swaps ``TopLeftCorner`` the same way.

Configuration
-------------

A jump/slon file, (see ``config.cx``). It has a ``default`` block, shallow-merged 
into every ``services+`` entry, and is re-read on every request; the services 
are rebuilt only when the parsed result actually changes.

There is no validation. Values are assumed present and of the right type.

Service keys:

    ``uid``         service id, first path segment, must be unique
    ``type``        one of the service types above
    ``crs``         the CRS this service speaks
    ``version``     WMS only, ``1.1.1`` or ``1.3.0``
    ``layer``       advertised layer name, default ``map``
    ``style``       advertised style name, default ``default``
    ``overlay``     draw a frame and a caption on every image
    ``overlayTile`` add the tile z/x/y to that caption

Tile grid keys (``xyz``, ``tms``, ``wmts``, ``wmts_rest``):

    ``tmsExtent``   the grid frame, an indexing frame, not a data extent
    ``tileSize``    pixels, normally 256
    ``resolutions`` units per pixel, one per level
    ``origin``      ``nw`` or ``sw``, defaults per service type

Scene keys, see below:

    ``extent``      the scene extent, always in EPSG:4326
    ``step``        graticule and checkerboard step in degrees
    ``graticule``, ``checker``, ``shapes``, ``labels``   which elements to draw
    ``labelSize``, ``labelPad``, ``labelPartials``, ``labelForce``
    ``background``  a CSS color, empty for a transparent background

The scene
---------

The scene is defined once in EPSG:4326 and reprojected by MapServer into
whatever a service speaks, so a projection error shows up as a distorted
graticule rather than as nothing at all. It consists of a densified graticule,
a checkerboard, a few reference shapes (circle, square, cross) and a point with
its own coordinates as a label at every graticule intersection. Lines are
densified so that reprojection curvature is visible; a two-point line would
reproject to a straight line and hide it.

``extent`` is the only place where the data location is configured. Each
service's advertised bounding box is derived from it, so moving the scene moves
every service. ``tmsExtent`` is separate on purpose: a tile grid frame follows
the CRS, not the data.

Labels are anchored to features, so they exercise the per-request placement a
tiled WMS client has to cope with: ``labelPartials`` and ``labelForce`` control
whether MapServer drops or duplicates a label at an image edge, and
``labelPad`` widens the text to find the point where a client's edge buffer
stops being enough.

Modules
-------

    ``server.py``    HTTP, routing, configuration loading and reloading
    ``services.py``  the service types, tile addressing, rendering and the
                     render cache
    ``scene.py``     the synthetic scene and the MapServer mapfile
    ``caps.cx``      jump templates for the capabilities documents
    ``config.cx``    example configuration

Rendering goes through ``gws.lib.mapserver``, with the scene built as a mapfile
of inline features. One MapServer quirk is worked around in ``scene.py``: an
image is drawn with square pixels and scaled to the requested size afterwards,
because MapServer silently widens an extent whose aspect does not match that
size.

Rendered images are cached in memory, keyed by scene, CRS, extent, size and
caption, so a repeated request costs nothing and load tests measure the client
rather than mockmap.
"""
