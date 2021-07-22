import yaml
from mapproxy.wsgiapp import make_wsgi_app

import gws
import gws.config.error
import gws.lib.json2
import gws.lib.os2

CONFIG_PATH = gws.CONFIG_DIR + '/mapproxy.yaml'
TMP_DIR = gws.TMP_DIR + '/mpx'


class _Config:
    def __init__(self):
        self.c = 0

        gws.ensure_dir(TMP_DIR)

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
            # https://mapproxy.org/docs/1.11.0/configuration.html#id14
            # "By default MapProxy assumes lat/long (north/east) order for all geographic and x/y (east/north) order for all projected SRS."
            # we need to change that because our extents are always x/y (lon/lat) even if a CRS says otherwise
            'srs': {
                'axis_order_en': ['EPSG:4326']
            },
            'cache': {
                'base_dir': gws.MAPPROXY_CACHE_DIR,
                'lock_dir': TMP_DIR + '/locks_' + gws.random_string(16),
                'tile_lock_dir': TMP_DIR + '/tile_locks_' + gws.random_string(16),
                'concurrent_tile_creators': 1,
                'max_tile_limit': 5000,

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

        uid = kind + '_' + gws.lib.json2.to_hash(c)

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


def create(root: gws.RootObject):
    mc = _Config()

    for layer in root.find_all('gws.ext.layer'):
        m = getattr(layer, 'mapproxy_config')
        if m:
            m(mc)

    cfg = mc.as_dict()
    if not cfg.get('layers'):
        return

    crs = set()
    for p in root.find_all('gws.base.map'):
        crs.add(gws.get(p, 'crs'))
    for p in root.find_all('gws.ext.ows.service'):
        crs.update(gws.get(p, 'supported_crs', default=[]))
    cfg['services']['wms']['srs'] = sorted(crs)

    return cfg


def create_and_save(root: gws.RootObject):
    cfg = create(root)
    if not cfg:
        gws.log.warn('mapproxy: NO CONFIG')
        gws.lib.os2.unlink(CONFIG_PATH)
        return

    cfg_str = yaml.dump(cfg)

    # make sure the config is ok before starting the server!
    test_path = CONFIG_PATH + '.test.yaml'
    gws.write_file(test_path, cfg_str)

    try:
        make_wsgi_app(test_path)
    except Exception as e:
        raise gws.config.error.ConfigurationError(f'MAPPROXY ERROR: {e!r}') from e

    gws.lib.os2.unlink(test_path)

    # write into the real config path
    gws.write_file(CONFIG_PATH, cfg_str)
