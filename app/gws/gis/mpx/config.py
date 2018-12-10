import yaml
from mapproxy.wsgiapp import make_wsgi_app

import gws
import gws.config
import gws.tools.shell as sh


# import gws.gis.proj

# def _transform_bbox(self, c):
#     # prevent mp from transforming the same bbox over and over again
#
#     if 'bbox' in c:
#         src_crs = c.get('bbox_srs') or c.get('srs')
#         dst_crs = 'EPSG:4326'
#         c['bbox'] = list(gws.gis.proj.transform_bbox(c['bbox'], src_crs, dst_crs))
#         c['bbox_srs'] = dst_crs


class _Config:
    def __init__(self):
        self.c = 0

        self.d = {
            'sources': {},
            'grids': {},
            'caches': {},
            'layers': [],
            'services': {
                'wms': {
                    'image_formats': ['image/png'],
                    'max_output_pixels': [9000, 9000]
                },
                'wmts': {
                    'kvp': True,
                    'restful': False
                }
            },

            'globals': {
                'cache': {
                    'base_dir': gws.MAPPROXY_CACHE_DIR,
                    'lock_dir': '/tmp/locks_' + gws.random_string(16),
                    'tile_lock_dir': '/tmp/tile_locks_' + gws.random_string(16),
                    'concurrent_tile_creators': 4,

                },
                'image': {
                    'resampling_method': 'bicubic',
                    'stretch_factor': 1.15,
                    'max_shrink_factor': 4.0,
                    'paletted': False,
                    'formats': {
                        'image/png': {
                            'mode': 'RGBA',
                            'transparent': True,
                        }
                    }
                }
            },
        }

    def _add(self, kind, obj, c, uid):
        # mpx doesn't like tuples
        if 'bbox' in c and isinstance(c['bbox'], tuple):
            c['bbox'] = list(c['bbox'])
        uid = '%s_%s' % (kind, uid or obj.uid)
        self.d[kind + 's'][uid] = c
        return uid

    def cache(self, obj, c, uid=None):
        return self._add('cache', obj, c, uid)

    def source(self, obj, c, uid=None):
        return self._add('source', obj, c, uid)

    def grid(self, obj, c, uid=None):
        # self._transform_bbox(c)
        return self._add('grid', obj, c, uid)

    def layer(self, obj, c, uid=None):
        uid = uid or obj.uid
        self.d['layers'].append(gws.extend(c, {'name': uid}))
        return uid

    def as_dict(self):
        return self.d


def create():
    mc = _Config()

    for r in gws.config.find_all('gws.ext.gis.layer'):
        r.mapproxy_config(mc)

    cfg = mc.as_dict()
    if not cfg['layers']:
        return

    crs = set(map.crs for map in gws.config.find_all('gws.common.map'))
    cfg['services']['wms']['srs'] = sorted(crs)

    return cfg


def create_and_save(path):
    test_path = path + '.test.yaml'
    sh.unlink(test_path)

    cfg = create()
    if not cfg:
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
