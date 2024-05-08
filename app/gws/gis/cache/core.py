"""Cache management."""

import datetime
import math
import os
import re

import gws
import gws.config
import gws.gis.mpx.config
import gws.lib.osx


class Config(gws.Config):
    """Global cache options"""

    seedingMaxTime: gws.Duration = '600'
    """max. time for a seeding job"""
    seedingConcurrency: int = 1
    """number of concurrent seeding jobs"""


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
    layers: list[gws.Node]
    mpxCache: dict
    grids: dict[int, Grid]
    config: dict
    counts: dict
    dirname: str


class Status(gws.Data):
    entries: list[Entry]
    staleDirs: list[str]


def status(root: gws.Root, layer_uids=None, with_counts=True) -> Status:
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
    s = status(root, with_counts=False)
    for d in s.staleDirs:
        _remove_dir(d)


def drop(root: gws.Root, layer_uids=None):
    s = status(root, layer_uids=layer_uids, with_counts=False)
    for e in s.entries:
        _remove_dir(e.dirname)


def seed(root: gws.Root, layer_uids=None, max_time=None, concurrency=1, levels=None):
    pass


#     mpx_config = gws.gis.mpx.config.create(root)
#     entries = _entries(root, mpx_config, layer_uids)
#
#
#     seeds = {}
#
#     for layer, cc in _cached_layers(root, mpx_config, layer_uids):
#         seeds[cc['uid']] =
#
#     ts = datetime.datetime.now() - datetime.timedelta(seconds=layer.cache.maxAge)
#
#     return {
#         'caches': [cc['name']],
#         'grids': cc['grids'],
#         'levels': levels or list(range(layer.cache.maxLevel)),
#         'refresh_before': {
#             'time': ts.strftime("%Y-%m-%dT%H:%M:%S")
#         }
#     }
#
#     if not seeds:
#         return True
#
#     path = gws.c.CONFIG_DIR + '/mapproxy.seed.yaml'
#     cfg = {
#         'seeds': seeds
#     }
#
#     with open(path, 'wt') as fp:
#         fp.write(yaml.dump(cfg))
#
#     cmd = [
#         '/usr/local/bin/mapproxy-seed',
#         '-f', gws.c.CONFIG_DIR + '/mapproxy.yaml',
#         '-c', str(concurrency),
#         path
#     ]
#     try:
#         gws.lib.osx.run(cmd, echo=True, timeout=max_time)
#     except gws.lib.osx.TimeoutError:
#         return False
#     except KeyboardInterrupt:
#         return False
#
#     return True


def store_in_web_cache(url: str, img: bytes):
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


def _enum_entries(root: gws.Root, mpx_config, layer_uids=None):
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


def _remove_dir(dirname):
    cmd = ['rm', '-fr', dirname]
    gws.lib.osx.run(cmd, echo=True)
    gws.log.info(f'removed {dirname}')
