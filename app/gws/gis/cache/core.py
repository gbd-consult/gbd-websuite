"""Tile cache inventory, filtering and maintenance."""

import os
from typing import Optional

import gws
import gws.lib.grid
import gws.lib.osx as osx


class LayerConfig(gws.Config):
    """Tile cache settings of a layer."""

    name: str = ''
    """Cache directory name. (added in 8.5)"""
    maxAge: gws.Duration = '7d'
    """Max. age of cached tiles."""
    maxLevel: int = 18
    """Finest zoom level to cache. (changed in 8.5)"""
    requestBuffer: int = 64
    """Pixel buffer around source requests, to keep labels consistent across tiles. (changed in 8.5)"""
    requestTiles: int = 4
    """Tiles per side of the block rendered in one source request. (changed in 8.5)"""
    crs: Optional[list[gws.CrsName]]
    """CRS to cache tiles in. (added in 8.5)"""


class GlobalConfig(gws.Config):
    """Global tile cache options."""

    seedingMaxTime: gws.Duration = '10m'
    """Time limit for a cache seeding run."""
    seedingConcurrency: int = 1
    """Number of parallel threads for cache seeding."""


class Level(gws.Data):
    """A zoom level of a cache, with statistics."""

    z: int
    """Zoom level."""
    gridRange: gws.MapTileRange
    """Tile range of the level, possibly restricted by a filter."""
    gridSize: gws.Size
    """Number of tile columns and rows in ``gridRange``."""
    resolution: float
    """Resolution of the level in map units per pixel."""
    seedTime: float
    """Seconds spent seeding this level."""
    cachedTiles: int
    """Number of tiles already in the store."""
    failedTiles: int
    """Number of tiles that failed during seeding."""
    fetchedTiles: int
    """Number of tiles fetched during seeding."""
    totalTiles: int
    """Number of tiles in ``gridRange``."""
    percentCached: int
    """Percentage of cached tiles."""
    cachedRange: Optional[gws.MapTileRange]
    """Bounding tile range of the stored tiles."""
    fileSize: int
    """Total size of the stored tiles in bytes."""


class Cache(gws.Data):
    """A tile cache, shared by all layers with the same cache name."""

    name: str
    """Cache name, also the name of the cache directory."""
    dir: str
    """Path to the cache directory, empty if it does not exist."""
    grabber: gws.Grabber
    """Grabber of the first layer that uses this cache."""
    layers: list[gws.Layer]
    """Layers that use this cache."""
    layerTitle: str
    """Title of the first layer."""
    layerType: str
    """Type of the first layer."""
    levels: list[Level]
    """Zoom levels of the cache, up to the cache max. level."""
    seedStatus: str
    """Status of the last seeding run, e.g. ``timeout``, empty if it completed."""
    cachedTiles: int
    """Number of stored tiles."""
    fileSize: int
    """Total size of the stored tiles in bytes."""


class Inventory(gws.Data):
    """All tile caches of a configuration."""

    caches: list[Cache]
    """Configured caches."""
    orphanDirs: list[str]
    """Directories in the cache directory that belong to no configured cache."""


class Filter(gws.Data):
    """Selects caches, levels and tiles from an inventory. Empty fields select everything."""

    layerUids: list[str]
    """Select caches used by any of these layers."""
    cacheNames: list[str]
    """Select caches whose names start with any of these prefixes."""
    srids: list[int]
    """Select caches in these CRS."""
    levels: list[int]
    """Select these zoom levels."""
    bbox: Optional[gws.Bounds]
    """Select tiles in this area, which must be in the CRS of the caches."""


class SeedOptions(gws.Data):
    """Options for a seeding run."""

    filter: Filter
    """Selects caches, levels and tiles to seed."""
    maxTime: int
    """Time limit in seconds."""
    concurrency: int
    """Number of worker threads."""
    maxAge: Optional[int]
    """Refetch tiles older than this (seconds), capped by the cache max. age."""


class SeedResult(gws.Data):
    """Result of a seeding run."""

    caches: list[Cache]
    """Seeded caches, with per-level statistics."""
    seedTime: float
    """Total run time in seconds."""
    seedStatus: str
    """Status of the run: empty if completed, ``timeout``, ``interrupted`` or ``locked``."""


def inventory(root: gws.Root) -> Inventory:
    """Collect all tile caches of a configuration.

    Caches are taken from the grabbers of all layers, skipping grabbers with a zero max. age.
    Each cache gets its levels up to the cache max. level and the existing cache directory.
    Directories in the cache directory that belong to no cache are listed as orphans.
    Statistics are not computed here, see ``add_stats``.

    Args:
        root: Configuration root.

    Returns:
        The inventory, caches sorted by layer type, layer title and name.
    """

    inv = Inventory(caches=[], orphanDirs=[])
    cmap = {}

    for la in root.find_all(gws.ext.object.layer):
        for gr in getattr(la, 'grabbers', {}).values():
            if gr.cache.maxAge <= 0:
                continue
            if gr.cache.name not in cmap:
                cmap[gr.cache.name] = Cache(
                    name=gr.cache.name,
                    grabber=gr,
                    dir='',
                    layers=[],
                    layerTitle=getattr(la, 'title', ''),
                    layerType=la.extType,
                    levels=[],
                    seedStatus='',
                    cachedTiles=0,
                    fileSize=0,
                )
            cmap[gr.cache.name].layers.append(la)

    inv.caches = list(cmap.values())
    inv.caches.sort(key=lambda c: (c.layerType, c.layerTitle, c.name))

    for c in inv.caches:
        for z in c.grabber.levels():
            if z > c.grabber.cache.maxLevel:
                break
            mtr = c.grabber.tile_range_for_level(z)
            nx = mtr[2] - mtr[0] + 1
            ny = mtr[3] - mtr[1] + 1
            c.levels.append(
                Level(
                    z=z,
                    gridRange=mtr,
                    gridSize=(nx, ny),
                    resolution=gws.lib.grid.resolution_for_level(c.grabber.grid, z),
                    seedTime=0,
                    cachedTiles=0,
                    failedTiles=0,
                    fetchedTiles=0,
                    totalTiles=nx * ny,
                    percentCached=0,
                    fileSize=0,
                )
            )

    for de in osx.find_entries(gws.c.MAP_CACHE_DIR, deep=False):
        if not de.is_dir():
            continue
        c = cmap.get(de.name)
        if not c:
            inv.orphanDirs.append(de.path)
            continue
        c.dir = de.path

    return inv


def apply_filter(inv: Inventory, flt: Filter):
    """Filter the inventory in place.

    Selects caches by layer, cache name prefix and CRS, then restricts their levels to ``flt.levels``
    and to the tile ranges covering ``flt.bbox``, which must be in the CRS of the caches.
    Caches without levels are removed.

    Args:
        inv: Inventory to filter.
        flt: Filter.
    """

    caches = []

    for c in inv.caches:
        b1 = not flt.layerUids or any(la.uid in flt.layerUids for la in c.layers)
        b2 = not flt.cacheNames or any(c.name.startswith(cn) for cn in flt.cacheNames)
        b3 = not flt.srids or c.grabber.targetCrs.srid in flt.srids
        if not (b1 and b2 and b3):
            continue

        levels = []
        for lv in c.levels:
            if flt.levels and lv.z not in flt.levels:
                continue
            if flt.bbox:
                mtr = gws.lib.grid.range_for_extent(c.grabber.grid, flt.bbox.extent, lv.z)
                mtr = gws.lib.grid.intersect_ranges(lv.gridRange, mtr) if mtr else None
                if not mtr:
                    continue
                lv.gridRange = mtr
                lv.gridSize = (mtr[2] - mtr[0] + 1, mtr[3] - mtr[1] + 1)
                lv.totalTiles = lv.gridSize[0] * lv.gridSize[1]
            levels.append(lv)

        if levels:
            c.levels = levels
            caches.append(c)

    inv.caches = caches


def add_stats(inv: Inventory):
    """Read store statistics into the inventory.

    Sets the number and size of stored tiles and the percentage cached for each level and cache.

    Args:
        inv: Inventory to update.
    """

    for c in inv.caches:
        for lv in c.levels:
            s = c.grabber.store.stats_for_level(lv.z)
            lv.cachedTiles = s.count
            lv.fileSize = s.size
            lv.cachedRange = s.range
            lv.percentCached = percent_cached(lv)
        c.fileSize = sum(lv.fileSize for lv in c.levels)
        c.cachedTiles = sum(lv.cachedTiles for lv in c.levels)


def percent_cached(lv: Level) -> int:
    """Compute the percentage of cached and fetched tiles of a level.

    Args:
        lv: Level.

    Returns:
        A percentage from 0 to 100, at least 1 if any tile is present.
    """

    n = lv.cachedTiles + lv.fetchedTiles
    if not n or not lv.totalTiles:
        return 0
    return min(100, max(1, int(n * 100 / lv.totalTiles)))


def percentage_by_level(c: Cache) -> list[int]:
    """Compute the cached percentages of all levels of a cache.

    Args:
        c: Cache.

    Returns:
        A list of percentages indexed by level, up to the cache max. level.
        Levels not in the cache are 0.
    """

    ps = [0] * (c.grabber.cache.maxLevel + 1)
    for lv in c.levels:
        ps[lv.z] = percent_cached(lv)
    return ps


def cleanup(root: gws.Root):
    """Remove orphan cache directories.

    Args:
        root: Configuration root.
    """

    inv = inventory(root)
    for d in inv.orphanDirs:
        gws.log.info(f'cleanup: removing orphan cache directory {d}')
        osx.rmdir(d)


def drop(root: gws.Root, flt: Optional[Filter] = None):
    """Remove cached tiles.

    If the filter has neither levels nor a bbox, the whole store of each selected cache is removed.
    Otherwise, the tile ranges covering the bbox, or the selected levels, are removed.

    Args:
        root: Configuration root.
        flt: Filter, selects all caches if omitted.
    """

    flt = flt or Filter()
    inv = inventory(root)
    apply_filter(inv, flt)

    for c in inv.caches:
        if not flt.bbox and not flt.levels:
            if c.dir:
                gws.log.info(f'drop: removing cache directory {c.dir}')
            c.grabber.store.drop()
            continue
        for lv in c.levels:
            if flt.bbox:
                gws.log.info(f'drop: {c.name}: removing tiles {lv.gridRange}')
                c.grabber.store.drop_range(lv.gridRange)
            else:
                gws.log.info(f'drop: {c.name}: removing level {lv.z}')
                c.grabber.store.drop_level(lv.z)


def store_in_web_cache(url: str, img: bytes):
    """Store an image in the web cache.

    Writes the image to ``FASTCACHE_DIR`` under the URL path. Write errors are logged and ignored.

    Args:
        url: URL path, used as the file path in the web cache.
        img: Image data.
    """
    path = gws.c.FASTCACHE_DIR + url
    try:
        os.makedirs(os.path.dirname(path), 0o755, exist_ok=True)
        gws.u.write_file_b(path, img)
    except OSError:
        gws.log.warning(f'store_in_web_cache FAILED path={path!r}')
