import datetime
import math
import os
import re

import gws
import gws.config
import gws.gis.mpx.config
import gws.lib.osx
import gws.types as t


class Config(gws.Config):
    """Global cache options"""

    seedingMaxTime: gws.Duration = '600' 
    """max. time for a seeding job"""
    seedingConcurrency: int = 1 
    """number of concurrent seeding jobs"""


# cache API

class Entry(gws.Data):
    uid: str
    layers: list[gws.INode]
    mpx_cache: dict
    mpx_grids: list[dict]
    config: dict
    counts: dict
    dirname: str


def status(root: gws.IRoot, layer_uids=None, with_counts=True):
    mpx_config = gws.gis.mpx.config.create(root)
    entries = _entries(root, mpx_config, layer_uids)

    if with_counts:
        _update_file_counts(entries)

    dirs = list(gws.lib.osx.find_files(gws.MAPPROXY_CACHE_DIR, pattern=r'^\w+$', deep=False))
    dangling_dirs = [d for d in dirs if not any(d.startswith(e.uid) for e in entries)]

    return {
        'entries': entries,
        'dangling_dirs': dangling_dirs,
    }


def seed(root: gws.IRoot, layer_uids=None, max_time=None, concurrency=1, levels=None):
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
#     path = gws.CONFIG_DIR + '/mapproxy.seed.yaml'
#     cfg = {
#         'seeds': seeds
#     }
#
#     with open(path, 'wt') as fp:
#         fp.write(yaml.dump(cfg))
#
#     cmd = [
#         '/usr/local/bin/mapproxy-seed',
#         '-f', gws.CONFIG_DIR + '/mapproxy.yaml',
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
    path = gws.WEB_CACHE_DIR + url
    dirname = os.path.dirname(path)
    tmp = dirname + '/' + gws.random_string(64)
    try:
        os.makedirs(dirname, 0o755, exist_ok=True)
        gws.write_file_b(tmp, img)
        os.rename(tmp, path)
    except OSError:
        gws.log.warning(f'store_in_web_cache FAILED path={path!r}')


def _update_file_counts(entries):
    files = list(gws.lib.osx.find_files(gws.MAPPROXY_CACHE_DIR))

    # file_counts = {}
    #
    # for path in files:
    #     for e in entries:
    #         if f.startswith(e.dirname):
    #             # we use the mp layout all the way: zz/xxxx/xxxx/yyyy/yyyy.format
    #             m = re.search(r'(\d+)/(\d+)/(\d+)/(\d+)/(\d+)\.png$', path)
    #             z0, x1, x2, y1, y2 = m.groups()
    #             x = int(x1) * 1000 + int(x2)
    #             y = int(y1) * 1000 + int(y2)
    #             z = int(z0)
    #
    #
    #
    #
    #
    #         x, y, z = _path_to_xyz(f)
    #         file_counts[z] += 1
    #
    # out = []
    #
    # for d in _calc_grids(cc['grid']):
    #     d['num_files'] = file_counts[d['z']]
    #     out.append(d)
    #
    # return out


def _entries(root: gws.IRoot, mpx_config, layer_uids=None):
    caches: gws.Dict[str, Entry] = {}

    for layer in root.find_all('gws.ext.layer'):

        if layer_uids and layer.uid not in layer_uids:
            continue

        for uid, cc in mpx_config['caches'].items():
            if cc.get('disable_storage') or gws.get(layer, 'cache_uid') != uid:
                continue

            if uid in caches:
                caches[uid].layers.append(layer)
                continue

            cfg = gws.get(layer, 'cache')
            grids = [mpx_config['grids'][g] for g in cc['grids']]
            crs = grids[0]['srs'].replace(':', '')

            caches[uid] = Entry(
                uid=uid,
                layers=[layer],
                mpx_cache=cc,
                mpx_grids=grids,
                config=vars(cfg) if cfg else {},
                counts={},
                dirname=f'{gws.MAPPROXY_CACHE_DIR}/{uid}_{crs}',
            )

    return list(caches.values())


# see _calc_grids in mapproxy/grid.py

def _calc_grids(grid):
    bbox = grid['bbox']
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    ts = grid['tile_size']
    for z, res in enumerate(grid['res']):
        yield {
            'z': z,
            'res': res,
            'maxx': max(math.ceil(w // res / ts[0]), 1),
            'maxy': max(math.ceil(h // res / ts[1]), 1),
        }


def _remove_dir(dirname):
    cmd = ['rm', '-fr', dirname]
    gws.lib.osx.run(cmd, echo=True)
    gws.log.info(f'removed {dirname}')


# CLI tool

# class SeedParams(gws.CliParams):
#     layers: list[str] 
#     """list of layer IDs"""
#     levels: list[int] 
#     """zoom levels to build the cache for"""
#
#
# class StatusParams(gws.CliParams):
#     layers: t.Optional[list[str]] 
#     """list of layer IDs"""
#
#
# gws.ext.new.cli('cache')

# class Cli(gws.Node):
#
#     @gws.ext.command()
#     def status(self, p: StatusParams):
#         """Display the cache status"""
#
#         root = gws.config.loader.load()
#
#         st = status(root, p.layers)
#
#         if not st:
#             print('no cached layers found')
#             return
#
#         for laUid, info in sorted(st.items()):
#             print()
#             print(f"CACHE  : {info['cache_uid']}")
#             print(f"CONFIG : {repr(info['config'])}")
#             print(f"LAYER  :")
#             for uid in info['layer_uids']:
#                 print(f"    {uid}")
#             print()
#
#             data = []
#
#             for r in info['counts']:
#                 data.append({
#                     'zoom': r['z'],
#                     'scale': round(units.res_to_scale(r['res'])),
#                     'grid': str(r['maxx']) + 'x' + str(r['maxy']),
#                     'total': r['maxx'] * r['maxy'],
#                     'cached': r['num_files'],
#                     '%%': int(100 * (r['num_files'] / (r['maxx'] * r['maxy']))),
#                 })
#             print(clihelpers.text_table(data, ['zoom', 'scale', 'grid', 'total', 'cached', '%%']))
#
#         print()
#         print()
#
#         u = gws.gis.cache.dangling_dirs(root)
#         if u:
#             print(f'{len(u)} DANGLING CACHES:')
#             print()
#             print('\n'.join(u))
#
#         print()
#
#
# def clean(root: gws.IRoot):
#     st = status(root)
#     for d in st['dangling_dirs']:
#         _remove_dir(gws.MAPPROXY_CACHE_DIR + '/' + d)
#
#
# def drop(root: gws.IRoot, layer_uids=None):
#     mc = gws.gis.mpx.config.create(root)
#
#     for layer, cc in _cached_layers(root, mc, layer_uids):
#         dirname = _dirname_for_cache(cc)
#         if os.path.isdir(dirname):
#             _remove_dir(dirname)
#
#
#
#
# @arg('--layers', help='comma separated list of layer IDs')
# def clean():
#     """Clean up the cache."""
#
#     root = gws.config.loader.load()
#     gws.gis.cache.clean(root)
#
#
# @arg('--layers', help='comma separated list of layer IDs')
# def drop(layers=None):
#     """Drop caches for specific or all layers."""
#
#     root = gws.config.loader.load()
#     if layers:
#         layers = _as_list(layers)
#     gws.gis.cache.drop(root, layers)
#
#
# _SEED_LOCKFILE = gws.CONFIG_DIR + '/mapproxy.seed.lock'
#
#
# @arg('--layers', help='comma separated list of layer IDs')
# @arg('--levels', help='zoom levels to build the cache for')
# def seed(layers=None, levels=None):
#     """Start the cache seeding process"""
#
#     root = gws.config.loader.load()
#
#     if layers:
#         layers = _as_list(layers)
#
#     st = gws.gis.cache.status(root, layers)
#
#     if not st:
#         print('no cached layers found')
#         return
#
#     if levels:
#         levels = [int(x) for x in _as_list(levels)]
#
#     with gws.lib.misc.lock(_SEED_LOCKFILE) as ok:
#         if not ok:
#             print('seed already running')
#             return
#
#         max_time = root.app.cfg('seeding.maxTime')
#         concurrency = root.app.cfg('seeding.concurrency')
#         ts = time.time()
#
#         print(f'\nSTART SEEDING (maxTime={max_time} concurrency={concurrency}), ^C ANYTIME TO CANCEL...\n')
#         done = gws.gis.cache.seed(root, layers, max_time, concurrency, levels)
#         print('=' * 40)
#         print('TIME: %.1f sec' % (time.time() - ts))
#         print(f'SEEDING COMPLETE' if done else 'SEEDING INCOMPLETE, PLEASE TRY AGAIN')
#
#
# def _as_list(s):
#     ls = []
#     for p in s.split(','):
#         p = p.strip()
#         if p:
#             ls.append(p)
#     return ls
