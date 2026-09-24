"""Cache management."""

import os
from typing import Optional, cast

import gws
from gws.base.grabber.___profile import stats
import gws.lib.grid
import gws.lib.osx as osx


class LayerConfig(gws.Config):
    """Layer cache configuration."""

    name: str = ''
    """Cache directory name. (added in 8.5)"""
    maxAge: gws.Duration = '7d'
    """Cache max. age."""
    maxLevel: int = 18
    """Max. zoom level to cache, on the global tile grid (18 is 0.6 m/px, about 1:2000). (changed in 8.5)"""
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


class Entry(gws.Data):
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


class Status(gws.Data):
    entries: list[Entry]
    staleDirs: list[str]


class Filter(gws.Data):
    layerUids: list[str]
    cacheNames: list[str]
    srids: list[int]
    bbox: Optional[gws.Bounds]


class SeedOptions(gws.Data):
    filter: Filter
    levels: list[int]
    maxTime: int
    concurrency: int
    maxAge: Optional[int]


class SeedResult(gws.Data):
    entries: list[Entry]
    seedTime: float
    seedStatus: str


def status(root: gws.Root) -> Status:
    st = Status(entries=[], staleDirs=[])
    emap = {}

    for la in root.find_all(gws.ext.object.layer):
        for gr in getattr(la, 'grabbers', {}).values():
            if gr.cache.maxAge <= 0:
                continue
            if gr.cache.name not in emap:
                emap[gr.cache.name] = Entry(
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
            emap[gr.cache.name].layers.append(la)

    st.entries = list(emap.values())
    st.entries.sort(key=lambda e: (e.layerType, e.layerTitle, e.name))

    for e in st.entries:
        for z in e.grabber.levels():
            if z > e.grabber.cache.maxLevel:
                break
            mtr = e.grabber.tile_range_for_level(z)
            nx = mtr[2] - mtr[0] + 1
            ny = mtr[3] - mtr[1] + 1
            e.levels.append(
                Level(
                    z=z,
                    gridRange=mtr,
                    gridSize=(nx, ny),
                    resolution=gws.lib.grid.resolution_for_level(e.grabber.grid, z),
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
        e = emap.get(de.name)
        if not e:
            st.staleDirs.append(de.path)
            continue
        e.dir = de.path

    return st


def apply_filter(st: Status, flt: Filter) -> Status:
    entries = []

    for e in st.entries:
        b1 = not flt.layerUids or any(la.uid in flt.layerUids for la in e.layers)
        b2 = not flt.cacheNames or any(e.name.startswith(cn) for cn in flt.cacheNames)
        b3 = not flt.srids or e.grabber.targetCrs.srid in flt.srids
        if b1 and b2 and b3:
            entries.append(e)

    return Status(entries=entries, staleDirs=st.staleDirs)


def restrict(st: Status, bbox: Optional[gws.Bounds] = None, levels: Optional[list[int]] = None):
    """Restrict status entries to the given levels and to the tile ranges covering a bbox.

    The bbox must be in the CRS of the entries. Levels outside the bbox and entries without levels are removed.
    """

    entries = []

    for e in st.entries:
        lvs = []
        for lv in e.levels:
            if levels and lv.z not in levels:
                continue
            if bbox:
                mtr = gws.lib.grid.range_for_extent(e.grabber.grid, bbox.extent, lv.z)
                mtr = gws.lib.grid.intersect_ranges(lv.gridRange, mtr) if mtr else None
                if not mtr:
                    continue
                lv.gridRange = mtr
                lv.gridSize = (mtr[2] - mtr[0] + 1, mtr[3] - mtr[1] + 1)
                lv.totalTiles = lv.gridSize[0] * lv.gridSize[1]
            lvs.append(lv)
        if lvs:
            e.levels = lvs
            entries.append(e)

    st.entries = entries


def add_counts_and_sizes(st: Status):
    for e in st.entries:
        for lv in e.levels:
            s = e.grabber.store.stats_for_level(lv.z)
            lv.cachedTiles = s.count
            lv.fileSize = s.size
            lv.cachedRange = s.range
            lv.percentCached = percent_cached(lv)
        e.fileSize = sum(lv.fileSize for lv in e.levels)
        e.cachedTiles = sum(lv.cachedTiles for lv in e.levels)


def percent_cached(lv: Level) -> int:
    """Percentage of cached and fetched tiles of a level, at least 1 if any tile is present."""

    n = lv.cachedTiles + lv.fetchedTiles
    if not n or not lv.totalTiles:
        return 0
    return min(100, max(1, int(n * 100 / lv.totalTiles)))


def percentage_by_level(e: Entry) -> list[int]:
    """Cached percentages of an entry, indexed by level, up to the max. cache level."""

    ps = [0] * (e.grabber.cache.maxLevel + 1)
    for lv in e.levels:
        ps[lv.z] = percent_cached(lv)
    return ps


def cleanup(root: gws.Root):
    st = status(root)
    for d in st.staleDirs:
        gws.log.info(f'cleanup: removing stale cache directory {d}')
        osx.rmdir(d)


def drop(root: gws.Root, flt: Optional[Filter] = None, levels: Optional[list[int]] = None):
    flt = flt or Filter()
    st = apply_filter(status(root), flt)

    if not flt.bbox and not levels:
        for e in st.entries:
            if e.dir:
                gws.log.info(f'drop: removing cache directory {e.dir}')
            e.grabber.store.drop()
        return

    restrict(st, flt.bbox, levels)

    for e in st.entries:
        for lv in e.levels:
            if flt.bbox:
                gws.log.info(f'drop: {e.name}: removing tiles {lv.gridRange}')
                e.grabber.store.drop_range(lv.gridRange)
            else:
                gws.log.info(f'drop: {e.name}: removing level {lv.z}')
                e.grabber.store.drop_level(lv.z)


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
