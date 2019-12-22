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
import gws.gis.mpx.config
import gws.common.layer

import gws.types as t


def status(layer_uids=None):
    mc = gws.gis.mpx.config.create()
    files = _get_files()

    st = {}

    layer: gws.common.layer.Image
    for layer, cc in _cached_layers(mc):
        uid = cc['name']
        if uid not in st:
            st[uid] = {
                'cache_uid': uid,
                'layer_uids': [layer.uid],
                'config': vars(layer.cache) if layer.cache else {},
                'mpx_config': cc,
                'counts': _file_counts_by_zoom_level(cc, mc, files)
            }
        else:
            st[uid]['layer_uids'].append(layer.uid)
            st[uid]['layer_uids'].sort()

    if not layer_uids:
        return st

    return {
        k: v
        for k, v in st.items()
        if any(uid in v['layer_uids'] for uid in layer_uids)
    }


def dangling_dirs():
    mc = gws.gis.mpx.config.create()
    used = {
        cc['name']
        for layer, cc in _cached_layers(mc)
    }
    return [
        d
        for d in _get_dirs()
        if not any(d.startswith(u) for u in used)
    ]


def clean():
    ds = []
    for d in dangling_dirs():
        _remove_dir(gws.MAPPROXY_CACHE_DIR + '/' + d)
    return len(ds)


def seed(layer_uids=None, max_time=None, concurrency=1, levels=None):
    mc = gws.gis.mpx.config.create()
    seeds = {}

    layer: gws.common.layer.Image
    for layer, cc in _cached_layers(mc, layer_uids):
        seeds[layer.cache_uid] = _seed_config(layer, cc, levels)

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


def drop(layer_uids=None):
    mc = gws.gis.mpx.config.create()

    for layer, cc in _cached_layers(mc, layer_uids):
        dirname = _dirname_for_cache(cc)
        if os.path.isdir(dirname):
            _remove_dir(dirname)


def store_in_web_cache(layer: t.LayerObject, x, y, z, img):
    dirname = gws.WEB_CACHE_DIR + f'/_/cmd/mapHttpGetXyz/layerUid/{layer.uid}/z/{z}/x/{x}/y/{y}'
    tmp = gws.random_string(64)
    try:
        os.makedirs(dirname, 0o755, exist_ok=True)
        with open(dirname + '/' + tmp, 'wb') as fp:
            fp.write(img)
        os.rename(dirname + '/' + tmp, dirname + '/t.png')
    except OSError:
        gws.log.warn(f'store_in_web_cache FAILED dir={dirname}')


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


def _file_counts_by_zoom_level(cc, mc, files):
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


def _cached_layers(mc, layer_uids=None):
    for layer in gws.config.root().find_all('gws.ext.layer'):
        cc = _cache_for_layer(t.cast(gws.common.layer.Image, layer), mc)
        if not cc:
            continue
        if layer_uids and layer.uid not in layer_uids:
            continue
        yield layer, cc


def _cache_for_layer(layer: gws.common.layer.Image, mc):
    for name, cc in mc['caches'].items():
        if layer.has_cache and name == layer.cache_uid and not cc['disable_storage']:
            return gws.extend(cc, {
                'name': name,
                'grid': mc['grids'][cc['grids'][0]],

            })


def _seed_config(layer, cc, levels):
    ts = datetime.datetime.now() - datetime.timedelta(seconds=layer.cache.maxAge)

    return {
        'caches': [cc['name']],
        'grids': cc['grids'],
        'levels': levels or list(range(layer.cache.maxLevel)),
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
    return list(misc.find_files(gws.MAPPROXY_CACHE_DIR))


def _get_dirs():
    ls = []

    for fname in os.listdir(gws.MAPPROXY_CACHE_DIR):
        if fname.startswith('.'):
            continue
        path = os.path.join(gws.MAPPROXY_CACHE_DIR, fname)
        if os.path.isdir(path):
            ls.append(fname)

    return ls


def _remove_dir(dirname):
    cmd = ['rm', '-fr', dirname]
    sh.run(cmd, echo=True)
    gws.log.info(f'removed {dirname}')
