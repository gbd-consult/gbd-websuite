"""Tile cache management.

This package holds the tile cache configuration, the filesystem tile store
used by grabbers, and the tools to inspect, seed and drop caches.

Tiles are produced and stored by grabbers (``gws.base.grabber``). Every
grabber of an image layer has a persistent store (``store.Object``) in a
directory under ``MAP_CACHE_DIR``, named after the grabber's cache name.
Layers with identical source bindings share one cache. This package works on
these stores from the outside: it collects all caches of a configuration
into an inventory, filters it and then reads statistics, removes tiles or
fills the cache by requesting tiles from the grabbers.

Submodules:

- ``core``: the config classes ``LayerConfig`` (per layer ``cache`` option)
  and ``GlobalConfig`` (application ``cache`` option), the inventory data
  structures (``Inventory``, ``Cache``, ``Level``, ``Filter``) and the
  functions to build, filter and drop an inventory, remove orphan cache
  directories and write to the web cache.
- ``store``: the filesystem tile store, which implements ``gws.TileStore``
  with the MapProxy ``mp`` directory layout
  (``<level>/<x div 10000>/<x mod 10000>/<y div 10000>/<y mod 10000>.<ext>``).
  A tile counts as stored while its file is younger than the max. age.
  Empty files and ``.tmp`` files are not counted in statistics.
- ``seed``: cache seeding. Missing tiles are requested from the grabbers in
  blocks, by a number of worker threads, until all are done or the time
  limit is reached. Only one seeding run can be active at a time.
- ``cli``: the ``gws cache`` command line commands ``status``,
  ``cleanup``, ``drop`` and ``seed``.

Example::

    gws cache status --details
    gws cache seed --layerUids my_layer --levels 0-12
    gws cache drop --crs 3857 --bbox 1000,2000,3000,4000 --levels 14

Configuration example, a layer cache and the global seeding options::

    map.layers+ {
        type "wms"
        provider.url "https://example.com/wms"
        cache { maxAge "30d" maxLevel 16 }
    }

    cache { seedingMaxTime "1h" seedingConcurrency 4 }

Python usage example::

    import gws.gis.cache.core as core

    inv = core.inventory(root)
    core.apply_filter(inv, core.Filter(cacheNames=['abc'], levels=[0, 1, 2]))
    core.add_stats(inv)
    for c in inv.caches:
        print(c.name, c.cachedTiles, core.percentage_by_level(c))
"""

from .core import (
    GlobalConfig,
    LayerConfig,
    store_in_web_cache,
)
from . import store

