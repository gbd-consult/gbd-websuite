"""Cache management."""

import os
from typing import Optional

import gws


class LayerConfig(gws.Config):
    """Layer cache configuration."""

    name: str = ''
    """Cache directory name; computed from the layer binding if omitted. (new in 8.5)"""
    maxAge: gws.Duration = '7d'
    """Cache max. age."""
    maxLevel: int = 6
    """Max. zoom level to cache."""
    requestBuffer: Optional[int]
    """Pixel buffer for source requests."""
    requestTiles: Optional[int]
    """Number of tiles to request at once."""


class GlobalConfig(gws.Config):
    """Global cache options"""

    seedingMaxTime: gws.Duration = '10m'
    """Max. time for a seeding job."""
    seedingConcurrency: int = 1
    """Number of concurrent seeding jobs."""


class Grid(gws.Data):
    uid: str
    z: int
    res: float
    maxX: int
    maxY: int
    totalTiles: int
    cachedTiles: int


class Entry(gws.Data):
    uid: str
    layers: list[gws.Layer]
    grids: dict[int, Grid]
    config: dict
    counts: dict
    dirname: str


class Status(gws.Data):
    entries: list[Entry]
    staleDirs: list[str]


def status(root: gws.Root, layer_uids=None, with_counts=True) -> Status:
    return Status(entries=[], staleDirs=[])


def cleanup(root: gws.Root):
    raise gws.Error('cache management is being redesigned')


def drop(root: gws.Root, layer_uids=None):
    raise gws.Error('cache management is being redesigned')


def seed(root: gws.Root, entries: list[Entry], levels: list[int], concurrency: int = 0):
    raise gws.Error('cache management is being redesigned')


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
