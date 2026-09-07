"""Cache management."""

import os
from typing import Optional, cast

import gws
import gws.lib.grid
import gws.lib.osx as osx


class LayerConfig(gws.Config):
    """Layer cache configuration."""

    name: str = ''
    """Cache directory name. (new in 8.5)"""
    maxAge: gws.Duration = '7d'
    """Cache max. age."""
    maxLevel: int = 6
    """Max. zoom level to cache."""
    requestBuffer: int = 64
    """Pixel buffer for source requests."""
    requestTiles: int = 4
    """Number of tiles to request at once."""

class GlobalConfig(gws.Config):
    """Global cache options"""

    seedingMaxTime: gws.Duration = '10m'
    """Max. time for a seeding job."""
    seedingConcurrency: int = 1
    """Number of concurrent seeding jobs."""


class Level(gws.Data):
    z: int
    gridSize: gws.Size
    resolution: float
    seedTime: float
    cachedTiles: int
    failedTiles: int
    fetchedTiles: int
    totalTiles: int


class Entry(gws.Data):
    name: str
    dir: str
    grabber: gws.Grabber
    layers: list[gws.Layer]
    levels: list[Level]
    seedStatus: str


class Status(gws.Data):
    entries: list[Entry]
    staleDirs: list[str]


class SeedOptions(gws.Data):
    layerUids: list[str]
    cacheNames: list[str]
    levels: list[int]
    maxTime: int
    concurrency: int


class SeedResult(gws.Data):
    entries: list[Entry]
    seedTime: float
    seedStatus: str


def status(root: gws.Root, layer_uids=None, cache_names=None, with_counts=True) -> Status:
    st = Status(entries=[], staleDirs=[])
    emap = {}

    for la in root.find_all(gws.ext.object.layer):
        gr = cast(gws.Grabber, getattr(la, 'grabber', None))
        if not gr or gr.cache.maxAge <= 0:
            continue
        if gr.cache.name not in emap:
            emap[gr.cache.name] = Entry(
                name=gr.cache.name,
                grabber=gr,
                dir='',
                layers=[],
                levels=[],
                seedStatus='',
            )
        emap[gr.cache.name].layers.append(la)

    for _, e in sorted(emap.items()):
        b1 = not layer_uids or any(la.uid in layer_uids for la in e.layers)
        b2 = not cache_names or any(e.name.startswith(cn) for cn in cache_names)
        if b1 and b2:
            st.entries.append(e)

    for e in st.entries:
        for z in e.grabber.levels():
            if z > e.grabber.cache.maxLevel:
                break
            x0, y0, x1, y1, _ = e.grabber.tile_range_for_level(z)
            e.levels.append(
                Level(
                    z=z,
                    gridSize=(x1 - x0 + 1, y1 - y0 + 1),
                    resolution=gws.lib.grid.resolution_for_level(e.grabber.grid, z),
                    seedTime=0,
                    cachedTiles=0,
                    failedTiles=0,
                    fetchedTiles=0,
                    totalTiles=(x1 - x0 + 1) * (y1 - y0 + 1),
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
        if with_counts and e in st.entries:
            for lv in e.levels:
                lv.cachedTiles = e.grabber.store.count_for_level(lv.z)

    return st


def cleanup(root: gws.Root):
    st = status(root, with_counts=False)
    for d in st.staleDirs:
        gws.log.info(f'cleanup: removing stale cache directory {d}')
        osx.rmdir(d)


def drop(root: gws.Root, layer_uids=None, cache_names=None):
    st = status(root, layer_uids=layer_uids, cache_names=cache_names, with_counts=False)
    for e in st.entries:
        gws.log.info(f'drop: removing cache directory {e.dir}')
        e.grabber.store.drop()


def store_in_web_cache(url: str, img: bytes):
    """Store an image in the web cache.

    Args:
        url: URL path to use as the cache key.
        img: Binary image data to store.

    Returns:
        None. Image is stored in the cache.
    """
    path = gws.c.FASTCACHE_DIR + url
    dirname = os.path.dirname(path)
    tmp = dirname + '/' + gws.u.random_string(64)
    try:
        os.makedirs(dirname, 0o755, exist_ok=True)
        gws.u.write_file_b(tmp, img)
        os.rename(tmp, path)
    except OSError:
        gws.log.warning(f'store_in_web_cache FAILED path={path!r}')
