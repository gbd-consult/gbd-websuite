import gws
import gws.common.layer
import gws.gis.util
import gws.gis.proj

import gws.types as t

from . import provider, types, util

class Config(gws.common.layer.VectorConfig, types.WfsServiceConfig):
    pass


class Object(gws.common.layer.Vector):
    def __init__(self):
        super().__init__()

        self.invert_axis_crs = []
        self.provider: provider.Object = None
        self.source_layers: t.List[types.SourceLayer] = []
        self.geometry_type = ''
        self.url = ''

    def configure(self):
        super().configure()

        util.configure_wfs(self)

        if not self.source_layers:
            raise gws.Error(f'no layers found in {self.uid!r}')

        self.source_layers = [self.source_layers[0]]
        self._add_default_search()

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.provider.meta,
            'sub_layers': self.source_layers
        }
        return self.description_template.render(context).content

    @property
    def props(self):
        return super().props.extend({
            'type': 'vector',
            ##'geometryType': self.geometry_type.upper(),
        })

    def get_features(self, bbox, limit=0):
        return util.find_features(self, bbox, self.map.crs, limit)

    def _add_default_search(self):
        p = self.var('search')
        if not p.enabled or p.providers:
            return

        cfg = {
            'type': 'wfs'
        }

        cfg_keys = [
            'capsCacheMaxAge',
            'invertAxis',
            'maxRequests',
            'sourceLayers',
            'url',
        ]

        for key in cfg_keys:
            cfg[key] = self.var(key)

        self.add_child('gws.ext.search.provider', t.Config(gws.compact(cfg)))
