import os
import collections
import datetime
import math
import re

import yaml

import gws
import gws.config
import gws.tools.misc as misc
import gws.tools.shell as sh
from .mpx import config


def status(layers=None):
    mc = config.create()
    files = _get_files()
    st = {}

    for layer, cc in _cached_layers(mc, layers):
        st[layer.uid] = _status_for_cache(cc, mc, files)

    return st


def seed(layers=None, max_time=None, concurrency=1, level=None):
    mc = config.create()
    seeds = {}

    for layer, cc in _cached_layers(mc, layers):
        seeds[layer.uid] = _seed_config(layer, cc, level)

    if not seeds:
        return False

    path = gws.CONFIG_DIR + '/mapproxy.seed.yaml'
    cfg = {
        'seeds': seeds
    }

    with open(path, 'wt') as fp:
        fp.write(yaml.dump(cfg))

    cmd = [
        '/usr/local/bin/mapproxy-seed',
        '-f', gws.CONFIG_DIR + '/mapproxy.yaml',
        '-c', str(concurrency),
        path
    ]
    try:
        sh.run(cmd, echo=True, timeout=max_time)
    except sh.TimeoutError:
        return False
    except KeyboardInterrupt:
        return False

    return True


def drop(layers=None):
    mc = config.create()
    for layer, cc in _cached_layers(mc, layers):
        dirname = _dirname_for_cache(cc)
        if os.path.isdir(dirname):
            cmd = ['rm', '-fr', dirname]
            sh.run(cmd, echo=True)
            gws.log.info(f'removed {dirname}')


def updateweb(layers):
    cmd = ['rm', '-fr', gws.WEB_CACHE_DIR + '/_']
    sh.run(cmd, echo=True)

    files = _get_files()
    mc = config.create()
    for layer, cc in _cached_layers(mc, layers):
        dirname = _dirname_for_cache(cc)
        for f in files:
            if f.startswith(dirname):
                x, y, z = _path_to_xyz(f)
                symlink_dir = gws.WEB_CACHE_DIR + f'/_/cmd/mapHttpGetXyz/layer/{layer.uid}/z/{z}/x/{x}/y/{y}'
                os.makedirs(symlink_dir, 0o755, exist_ok=True)
                os.symlink(f, symlink_dir + '/t.png')


def _path_to_xyz(path):
    # we use the mp layout all the way: zz/xxxx/xxxx/yyyy/yyyy.format
    m = re.search(r'(\d+)/(\d+)/(\d+)/(\d+)/(\d+)\.png$', path)
    z, x1, x2, y1, y2 = m.groups()
    return (
        int(x1) * 1000 + int(x2),
        int(y1) * 1000 + int(y2),
        int(z))


def _dirname_for_cache(cc):
    return gws.MAPPROXY_CACHE_DIR + '/' + cc['name'] + '_' + cc['grid']['srs'].replace(':', '')


def _status_for_cache(cc, mc, files):
    file_counts = collections.defaultdict(int)
    dirname = _dirname_for_cache(cc)
    for f in files:
        if f.startswith(dirname):
            x, y, z = _path_to_xyz(f)
            file_counts[z] += 1

    out = []

    for d in _calc_grids(cc['grid']):
        d['num_files'] = file_counts[d['z']]
        out.append(d)

    return out


def _cached_layers(mc, layers):
    for layer in gws.config.find_all('gws.ext.layer'):
        cc = _cache_for_layer(layer, mc)
        if not cc:
            continue
        if layers and layer.uid not in layers:
            continue
        yield layer, cc


def _cache_for_layer(layer, mc):
    for name, cc in mc['caches'].items():
        if name.startswith('cache_' + layer.uid) and name.endswith('_FRONT') and not cc['disable_storage']:
            return gws.extend(cc, {
                'name': name,
                'grid': mc['grids'][cc['grids'][0]],

            })


def _seed_config(layer, cc, level):
    ts = datetime.datetime.now() - datetime.timedelta(seconds=layer.cache.maxAge)

    if level is None:
        la, lz = 0, layer.cache.maxLevel
    else:
        la = lz = int(level)

    return {
        'caches': [cc['name']],
        'grids': cc['grids'],
        'levels': {'from': la, 'to': lz},
        'refresh_before': {
            'time': ts.strftime("%Y-%m-%dT%H:%M:%S")
        }
    }


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


def _get_files():
    # find /gws-var/cache/mpx  -type f -printf '%p %T@\n'

    return list(misc.find_files(gws.MAPPROXY_CACHE_DIR))
