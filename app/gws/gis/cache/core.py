"""Cache management."""

import math
import os
import re

import yaml

import gws
import gws.config
import gws.gis.mpx.config
import gws.lib.osx
import gws.lib.lock
import gws.lib.datetimex as datetimex

DEFAULT_MAX_TIME = 600
DEFAULT_CONCURRENCY = 1
DEFAULT_MAX_AGE = 7 * 24 * 3600
DEFAULT_MAX_LEVEL = 3


class Config(gws.Config):
    """Global cache options"""

    seedingMaxTime: gws.Duration = '600'
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
    mpxCache: dict
    grids: dict[int, Grid]
    config: dict
    counts: dict
    dirname: str


class Status(gws.Data):
    entries: list[Entry]
    staleDirs: list[str]


def status(root: gws.Root, layer_uids=None, with_counts=True) -> Status:
    """Retrieve cache status information.

    Args:
        root: Application root object.
        layer_uids: Optional list of layer UIDs to filter by.
        with_counts: Whether to include tile counts in the status.

    Returns:
        Status object containing cache entries and stale directories.
    """
    mpx_config = gws.gis.mpx.config.create(root)

    entries = []
    if mpx_config:
        entries = _enum_entries(root, mpx_config, layer_uids)

    if entries and with_counts:
        _update_file_counts(entries)

    all_dirs = list(gws.lib.osx.find_directories(gws.c.MAPPROXY_CACHE_DIR, deep=False))
    valid_dirs = set(e.dirname for e in entries)

    return Status(
        entries=entries,
        staleDirs=[d for d in all_dirs if d not in valid_dirs],
    )


def cleanup(root: gws.Root):
    """Remove stale cache directories.

    Args:
        root: Application root object.

    Returns:
        None. Stale directories are removed from the filesystem.
    """
    s = status(root, with_counts=False)
    for d in s.staleDirs:
        _remove_dir(d)


def drop(root: gws.Root, layer_uids=None):
    """Remove active cache directories.

    Args:
        root: Application root object.
        layer_uids: Optional list of layer UIDs to filter by.

    Returns:
        None. Cache directories are removed from the filesystem.
    """
    s = status(root, layer_uids=layer_uids, with_counts=False)
    for e in s.entries:
        _remove_dir(e.dirname)

PIXEL_PNG8 = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x03\x00\x00\x00(\xcb4\xbb\x00\x00\x00\x06PLTE\xff\xff\xff\x00\x00\x00U\xc2\xd3~\x00\x00\x00\x01tRNS\x00@\xe6\xd8f\x00\x00\x00\x0cIDATx\xdab`\x00\x080\x00\x00\x02\x00\x01OmY\xe1\x00\x00\x00\x00IEND\xaeB`\x82'

def seed(root: gws.Root, entries: list[Entry], levels: list[int]):
    """Generate and populate the cache for specified layers and zoom levels.

    Args:
        root: Application root object.
        entries: List of cache entries to seed.
        levels: List of zoom levels to generate cache for.

    Returns:
        None. Cache is populated with generated tiles.
    """
    # https://mapproxy.github.io/mapproxy/latest/seed.html#seeds
    seeds = {}

    for e in entries:
        cache_uid = e.uid

        c = e.layers[0].cache or gws.Data()
        max_age = c.get('maxAge') or DEFAULT_MAX_AGE
        max_level = c.get('maxLevel') or DEFAULT_MAX_LEVEL

        seeds[cache_uid] = dict(
            caches=[cache_uid],
            # grids=e.mpxCache['grids'],
            levels=levels or range(max_level + 1),
            refresh_before={
                'time': datetimex.to_iso_string(datetimex.to_utc(datetimex.add(seconds=-max_age)), with_tz=''),
            }
        )

    if not seeds:
        gws.log.info('no layers to seed')
        return

    lock_path = gws.c.CONFIG_DIR + '/mapproxy.seed.lock'

    with gws.lib.lock.SoftFileLock(lock_path) as ok:
        if not ok:
            gws.log.info('seeding already running')
            return

        mpx_config = gws.gis.mpx.config.create(root)
        mpx_config_path = gws.c.CONFIG_DIR + '/mapproxy.seed.main.yml'
        gws.u.write_file(mpx_config_path, yaml.dump(mpx_config))

        seed_config_path = gws.c.CONFIG_DIR + '/mapproxy.seed.yml'
        gws.u.write_file(seed_config_path, yaml.dump(dict(seeds=seeds)))

        max_time = root.app.cfg('cache.seedingMaxTime', default=DEFAULT_MAX_TIME)
        concurrency = root.app.cfg('cache.seedingConcurrency', default=DEFAULT_CONCURRENCY)

        # monkeypatch mapproxy to simply store an empty image in case of error
        empty_pixel_path = gws.c.CONFIG_DIR + '/mapproxy.seed.empty.png'
        gws.u.write_file_b(empty_pixel_path, PIXEL_PNG8)
        py = '/usr/local/lib/python3.10/dist-packages/mapproxy/client/http.py'
        s = gws.u.read_file(py)
        s = re.sub(r"raise HTTPClientError\('response is not an image.+", f'return ImageSource({empty_pixel_path!r})', s)
        gws.u.write_file(py, s)

        ts = gws.u.stime()
        gws.log.info(f'START SEEDING jobs={len(seeds)} {max_time=} {concurrency=}')
        gws.log.info(f'^C ANYTIME TO STOP...')

        cmd = f'''
            /usr/local/bin/mapproxy-seed
            -f {mpx_config_path}
            -c {concurrency}
            {seed_config_path}
        '''
        res = False
        try:
            gws.lib.osx.run(cmd, echo=True, timeout=max_time or DEFAULT_MAX_TIME)
            res = True
        except gws.lib.osx.TimeoutError:
            pass
        except KeyboardInterrupt:
            pass

        try:
            for p in gws.lib.osx.find_directories(gws.c.MAPPROXY_CACHE_DIR, deep=False):
                gws.lib.osx.run(f'chown -R {gws.c.UID}:{gws.c.GID} {p}', echo=True)
        except Exception as exc:
            gws.log.error('failed to chown cache dir: {exc!r}')

        gws.log.info(f'TIME: {gws.u.stime() - ts} sec')
        gws.log.info(f'SEEDING COMPLETE' if res else 'SEEDING INCOMPLETE, PLEASE TRY AGAIN')


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


def _update_file_counts(entries: list[Entry]):
    """Update cached tile counts for each entry.

    Args:
        entries: List of cache entries to update.

    Returns:
        None. The entries are updated in-place.
    """
    files = list(gws.lib.osx.find_files(gws.c.MAPPROXY_CACHE_DIR))

    for path in files:
        for e in entries:
            if path.startswith(e.dirname):
                # we use the mp layout all the way: zz/xxxx/xxxx/yyyy/yyyy.format
                m = re.search(r'(\d+)/(\d+)/(\d+)/(\d+)/(\d+)\.png$', path)
                z0, x1, x2, y1, y2 = m.groups()
                x = int(x1) * 1000 + int(x2)
                y = int(y1) * 1000 + int(y2)
                z = int(z0)
                g = e.grids.get(z)
                if g:
                    g.cachedTiles += 1


def _enum_entries(root: gws.Root, mpx_config: dict, layer_uids=None) -> list[Entry]:
    """Enumerate cache entries based on layer configuration.

    Args:
        root: Application root object.
        mpx_config: MapProxy configuration dictionary.
        layer_uids: Optional list of layer UIDs to filter by.

    Returns:
        List of cache Entry objects.
    """
    entries_map: dict[str, Entry] = {}

    for layer in root.find_all(gws.ext.object.layer):

        if layer_uids and layer.uid not in layer_uids:
            continue

        for uid, mpx_cache in mpx_config['caches'].items():
            if mpx_cache.get('disable_storage') or gws.u.get(layer, 'mpxCacheUid') != uid:
                continue

            if uid in entries_map:
                entries_map[uid].layers.append(layer)
                continue

            mpx_grids = [mpx_config['grids'][guid] for guid in mpx_cache['grids']]
            crs = mpx_grids[0]['srs'].replace(':', '')

            e = Entry(
                uid=uid,
                layers=[layer],
                mpxCache=mpx_cache,
                grids={},
                config={},
                dirname=f'{gws.c.MAPPROXY_CACHE_DIR}/{uid}_{crs}',
            )

            for g in mpx_grids:
                # see _calc_grids in mapproxy/grid.py
                bbox = g['bbox']
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                ts = g['tile_size']
                for z, res in enumerate(sorted(g['res'], reverse=True)):
                    maxx = max(math.ceil(w // res / ts[0]), 1)
                    maxy = max(math.ceil(h // res / ts[1]), 1)
                    e.grids[z] = Grid(
                        z=z,
                        res=res,
                        maxX=maxx,
                        maxY=maxy,
                        totalTiles=maxx * maxy,
                        cachedTiles=0,
                    )

            entries_map[uid] = e

    return list(entries_map.values())


def _remove_dir(dirname: str):
    """Remove a directory and its contents.

    Args:
        dirname: Path to the directory to remove.

    Returns:
        None. The directory is removed from the filesystem.
    """
    cmd = ['rm', '-fr', dirname]
    gws.lib.osx.run(cmd, echo=True)
    gws.log.info(f'removed {dirname}')