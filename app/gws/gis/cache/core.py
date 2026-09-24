"""Cache management."""

import os
from typing import Optional

import gws
import gws.lib.grid
import gws.lib.osx as osx


class LayerConfig(gws.Config):
    """Layer cache configuration."""

    name: str = ''
    """Cache directory name. (added in 8.5)"""
    maxAge: gws.Duration = '7d'
    """Cache max. age."""
    maxLevel: int = 18
    """Max. zoom level to cache, on the global tile grid. (changed in 8.5)"""
    requestBuffer: int = 64
    """Pixel buffer for source requests. (changed in 8.5)"""
    requestTiles: int = 4
    """Number of tiles to request at once. (changed in 8.5)"""
    crs: Optional[list[gws.CrsName]]
    """CRS to cache tiles in. By default, tiles are cached in all supported CRS. (added in 8.5)"""


class GlobalConfig(gws.Config):
    """Global cache options"""

    seedingMaxTime: gws.Duration = '10m'
    """Max. time for a seeding job."""
    seedingConcurrency: int = 1
    """Number of concurrent seeding jobs."""


class Level(gws.Data):
    z: int
    gridRange: gws.MapTileRange
    gridSize: gws.Size
    resolution: float
    seedTime: float
    cachedTiles: int
    failedTiles: int
    fetchedTiles: int
    totalTiles: int
    percentCached: int
    cachedRange: Optional[gws.MapTileRange]
    fileSize: int


class Cache(gws.Data):
    name: str
    dir: str
    grabber: gws.Grabber
    layers: list[gws.Layer]
    layerTitle: str
    layerType: str
    levels: list[Level]
    seedStatus: str
    cachedTiles: int
    fileSize: int


class Inventory(gws.Data):
    caches: list[Cache]
    orphanDirs: list[str]


class Filter(gws.Data):
    layerUids: list[str]
    cacheNames: list[str]
    srids: list[int]
    levels: list[int]
    bbox: Optional[gws.Bounds]


class SeedOptions(gws.Data):
    filter: Filter
    maxTime: int
    concurrency: int
    maxAge: Optional[int]


class SeedResult(gws.Data):
    caches: list[Cache]
    seedTime: float
    seedStatus: str


def inventory(root: gws.Root) -> Inventory:
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
    """Percentage of cached and fetched tiles of a level, at least 1 if any tile is present."""

    n = lv.cachedTiles + lv.fetchedTiles
    if not n or not lv.totalTiles:
        return 0
    return min(100, max(1, int(n * 100 / lv.totalTiles)))


def percentage_by_level(c: Cache) -> list[int]:
    """Cached percentages of a cache, indexed by level, up to the max. cache level."""

    ps = [0] * (c.grabber.cache.maxLevel + 1)
    for lv in c.levels:
        ps[lv.z] = percent_cached(lv)
    return ps


def cleanup(root: gws.Root):
    inv = inventory(root)
    for d in inv.orphanDirs:
        gws.log.info(f'cleanup: removing orphan cache directory {d}')
        osx.rmdir(d)


def drop(root: gws.Root, flt: Optional[Filter] = None):
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

    Args:
        url: URL path to use as the cache key.
        img: Binary image data to store.

    Returns:
        None. Image is stored in the cache.
    """
    path = gws.c.FASTCACHE_DIR + url
    try:
        os.makedirs(os.path.dirname(path), 0o755, exist_ok=True)
        gws.u.write_file_b(path, img)
    except OSError:
        gws.log.warning(f'store_in_web_cache FAILED path={path!r}')
