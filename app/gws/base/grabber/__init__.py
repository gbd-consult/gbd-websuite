"""Raster grabber.

A grabber provides raster images for a layer: it fetches them from a source,
aligns them to a tile grid, reprojects them into the target CRS, stores tiles
and reads them back. It runs in-process; there is no separate tile server,
no generated configuration and no HTTP between GWS and the tile machinery.

Grids
-----

There is one fixed tile grid per CRS, shared by all grabbers in that CRS
(``gws.lib.grid``). Projected CRS use the web mercator square as frame, one
256px tile at level 0 and the resolution ladder ``156543.03 / 2^z``;
geographic CRS use ``(-180, -90, 180, 90)``, two tiles at level 0 and
``0.703125 / 2^z`` degrees. The same level means the same ground resolution
in every projected CRS, and web mercator tile sources align 1:1 with the
EPSG:3857 grid. The ladder has no bottom: any resolution is served by
downsampling from the coarsest level that is not coarser than the target.
Levels beyond ``MAX_LEVEL`` (30) are not served.

Each grabber restricts the grid to the layer's extent: a tile range per level
(``tile_range_for_level``), derived from the layer's WGS extent clipped to
the area of use of the CRS. Tiles outside the range are transparent.

Grabbers and layers
-------------------

A layer holds one grabber per CRS the application supports (``layer.grabbers``,
keyed by SRID), created in ``post_configure_grabbers`` via the layer type's
``create_grabber``. A layer whose extent does not intersect a CRS gets no
grabber for it. Grabbers are plain objects, not nodes.

Concrete grabbers live next to their provider (``plugin/.../grabber.py``) and
extend one of two base classes:

- `box.Object` for sources that render arbitrary boxes (WMS, QGIS server,
  in-process MapServer for raster and MBTiles layers). Subclasses implement
  ``fetch_box_as_bytes`` and ``fetch_box_as_image``, one source request each.
- `tile.Object` for sources addressed as tile pyramids (tile services, WMTS).
  Subclasses provide ``sourceMatrices`` and ``fetch_tile_as_bytes``.

Fetching and composing
----------------------

The public API (`gws.Grabber`) is ``get_tile_as_bytes/image``,
``get_tiles_as_bytes/image_dict`` and ``get_box_as_bytes/image``. ``get``
may hit the store, ``compose`` assembles from parts and may issue several
source requests, ``fetch`` is exactly one source request. Both return forms
exist because encoding and decoding are not free; a caller asks for the
form it needs and never pays a round trip.

Box sources are meta-tiled: a block of ``requestTiles`` x ``requestTiles``
tiles is rendered as one request with a ``requestBuffer`` (pixels) around it,
then cut into tiles. This gives fewer source requests and consistent label
placement across tile seams. Requests larger than the provider's
``maxRequestPixels`` are split into chunks with the same buffer.

Tile sources fetch the source tiles covering a target tile at the best
matching source matrix (never upscaling), mosaic and warp them. Fetched
source tiles are kept in the ephemeral store, so neighbouring target tiles
reuse them. There is no meta-tiling for tile sources.

Reprojection is done by the base: source images are fetched in the source
CRS, clipped to the source CRS area of use, and warped into the target grid
with GDAL. A box outside the source area is transparent.

Boxes at a stored level are mosaicked from stored tiles; otherwise (uncached
layers, levels beyond ``cache.maxLevel``, dynamic requests) they are composed
directly at the exact resolution, which keeps print output free of pyramid
resampling.

Caching
-------

Every grabber has two tile stores with the same layout
(``gws.gis.cache.store``): the persistent store under
``MAP_CACHE_DIR/<cache name>_<srid>`` holds levels up to ``cache.maxLevel``
when the layer has ``withCache`` and a positive ``cache.maxAge``; everything
else goes to an ephemeral store under the ephemeral directory with a fixed
lifetime of ``EPHEMERAL_MAX_AGE`` (60 s), so that a block is composed once
per viewport even without a cache.

The cache name is a content hash of the source binding (provider, source
layers, style, format, extent, request shaping), or ``cache.name`` when
configured; layers with identical bindings share stores. ``cache.crs``
limits persistent caching to the listed CRS.

Block composition is locked on block identity (``gws.u.server_lock``), so
concurrent requests for tiles of one block trigger one source fetch. A lock
busy for longer than ``BLOCK_LOCK_TIMEOUT`` (60 s) yields the transparent tile
and a warning.

Dynamic requests
----------------

``params`` (e.g. the visible leaf set of a composite QGIS layer, passed by the
layer as ``lri.renderParams``) marks a request as dynamic. Dynamic requests
never touch the persistent store; they use an ephemeral store keyed by a
hash of the params and are meta-tiled and locked under the same key. Layers
must pass distilled params only, or every request goes uncached.

Configuration
-------------

On the layer:

- ``withCache`` (default true) and ``cache``: ``maxAge`` (7d), ``maxLevel``
  (18, a level of the global grid), ``requestTiles`` (4), ``requestBuffer``
  (64), ``crs``, ``name``.
- ``imageFormat``: the stored and returned format (default PNG8); JPEG and
  other opaque formats render transparency as a white background.
- ``display``: ``tile`` (grabber tiles), ``box`` (grabber boxes) or, for
  ``tile`` and ``wmts`` layers, ``client`` (the browser fetches source tiles
  directly; requires a source grid in the map CRS).

Per provider: ``maxRequestPixels`` for WMS (4096); MapServer-backed sources
use 9000.

Cache management is ``gws cache status | seed | drop`` (``gws.gis.cache``),
which works on the layers' grabbers and stores.

Notes and open issues
---------------------

- Source requests for the tiles covering one target tile are sequential;
  cross-CRS tile sources pay several upstream round trips per cold tile.
- Fetched source tiles stay in the ephemeral store for up to two hours,
  independent of the layer's cache settings.
- Meta-tiled composite QGIS layers need label placement that does not depend
  on the request extent (polygon labels: centroid of the whole polygon) and a
  ``requestBuffer`` at least as wide as the widest label, or labels differ
  across block seams.
- A source answering with an image of unexpected size raises
  ``gws.ExternalServiceError`` (empty tile, log entry) rather than being
  padded.
- The background color for opaque formats is not configurable.
- The own WMTS service composes a box per tile instead of reading stored
  tiles directly.
"""

from .core import Options, Object
from . import box, tile
