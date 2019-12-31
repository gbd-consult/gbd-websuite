import yaml
from mapproxy.wsgiapp import make_wsgi_app

import gws
import gws.config
import gws.tools.shell as sh
import gws.tools.json2

import gws.types as t

class _Config:
    def __init__(self):
        self.c = 0

        self.services = {
            'wms': {
                'image_formats': ['image/png'],
                'max_output_pixels': [9000, 9000]
            },
            'wmts': {
                'kvp': True,
                'restful': False
            }
        }

        self.globals = {
            'cache': {
                'base_dir': gws.MAPPROXY_CACHE_DIR,
                'lock_dir': gws.TMP_DIR + '/mpx/locks_' + gws.random_string(16),
                'tile_lock_dir': gws.TMP_DIR + '/mpx/tile_locks_' + gws.random_string(16),
                'concurrent_tile_creators': 1,

            },
            'image': {
                'resampling_method': 'bicubic',
                'stretch_factor': 1.15,
                'max_shrink_factor': 4.0,

                'formats': {
                    'png8': {
                        'format': 'image/png',
                        'mode': 'P',
                        'colors': 256,
                        'transparent': True,
                        'resampling_method': 'bicubic',
                    },
                    'png24': {
                        'format': 'image/png',
                        'mode': 'RGBA',
                        'colors': 0,
                        'transparent': True,
                        'resampling_method': 'bicubic',
                    }

                }
            }
        }

        self.cfg = {}

    def _add(self, kind, c):
        # mpx doesn't like tuples
        for k, v in c.items():
            if isinstance(v, tuple):
                c[k] = list(v)

        uid = kind + '_' + gws.tools.json2.to_hash(c)

        # clients might add their hash params starting with '$'
        c = {
            k: v
            for k, v in c.items()
            if not k.startswith('$')
        }

        self.cfg[uid] = {'kind': kind, 'c': c}
        return uid

    def _items(self, kind):
        for k, v in self.cfg.items():
            if v['kind'] == kind:
                yield k, v['c']

    def cache(self, c):
        return self._add('cache', c)

    def source(self, c):
        return self._add('source', c)

    def grid(self, c):
        # self._transform_extent(c)
        return self._add('grid', c)

    def layer(self, c):
        c['title'] = ''
        return self._add('layer', c)

    def as_dict(self):
        d = {
            'services': self.services,
            'globals': self.globals,
        }

        kinds = ['source', 'grid', 'cache', 'layer']
        for kind in kinds:
            d[kind + 's'] = {
                key: c
                for key, c in self._items(kind)
            }

        d['layers'] = sorted(d['layers'].values(), key=lambda x: x['name'])

        return d


def create(root: t.IRootObject):
    mc = _Config()

    r: t.ILayer
    for r  in root.find_all('gws.ext.layer'):
        r.mapproxy_config(mc)

    cfg = mc.as_dict()
    if not cfg['layers']:
        return

    m: t.IMap
    crs = set(m.crs for m in root.find_all('gws.common.map'))
    cfg['services']['wms']['srs'] = sorted(crs)

    return cfg


def create_and_save(root: t.IRootObject, path):
    test_path = path + '.test.yaml'
    sh.unlink(test_path)

    cfg = create(root)
    if not cfg:
        gws.log.warn('mapproxy: NO CONFIG')
        sh.unlink(path)
        return

    with open(test_path, 'wt') as fp:
        fp.write(yaml.dump(cfg))

    # make sure the config is ok before starting the server!
    try:
        make_wsgi_app(test_path)
    except Exception as e:
        raise gws.config.MapproxyConfigError(*e.args) from e

    sh.unlink(test_path)

    with open(path, 'wt') as fp:
        fp.write(yaml.dump(cfg))
