import re

import gws
import gws.config
import gws.config.parser
import gws.gis.layer
import gws.gis.source
import gws.gis.mpx as mpx
import gws.types as t


class Config(gws.gis.layer.ProxiedConfig):
    """tree layer"""
    pass


class LayerProps(gws.gis.layer.BaseProps):
    url: str
    layers: t.List[t.ext.gis.layer.LayerProps]


class Object(gws.gis.layer.Proxied):
    def __init__(self):
        super().__init__()
        self.root_layers: t.List[t.SourceLayer] = []
        self.name2sl = {}
        self.uid2leaf = {}
        self.layers: t.List[t.LayerObject] = []

    def configure(self):
        super().configure()

        if self.source.klass != 'gws.ext.gis.source.qgis':
            # see below in render
            raise ValueError('trees only work with qgis')

        self.root_layers = gws.gis.source.filter_layers(
            self.source.layers,
            self.var('sourceLayers')
        )
        if not self.root_layers:
            raise ValueError(f'no roots for layer "{self.uid}"')

        root_group = {
            'type': 'group',
            'title': '',
            'layers': gws.compact(self._layer(sl) for sl in self.root_layers)
        }

        root_cfg = gws.config.parser.parse(root_group, 'gws.ext.gis.layer.group.Config')
        for p in root_cfg.layers:
            self.layers.append(self.add_child('gws.ext.gis.layer', p))

        self._enum_leaves(self.children)

    def _enum_leaves(self, ls):
        for obj in ls:
            if obj.is_a('gws.ext.gis.layer.group'):
                self._enum_leaves(obj.children)
            elif obj.is_a('gws.ext.gis.layer.leaf'):
                obj.tree_layer = self
                obj.source_layer = self.name2sl[obj.wms_name]
                self.uid2leaf[obj.uid] = obj

    def _layer(self, sl: t.SourceLayer):
        if sl.is_group:
            layers = gws.compact(self._layer(la) for la in sl.layers)
            if not layers:
                return
            return {
                'type': 'group',
                'title': sl.meta.title,
                'clientOptions': {
                    'visible': sl.is_visible,
                    'expanded': sl.is_expanded,
                },
                'layers': layers
            }

        if not sl.name:
            return

        self.name2sl[sl.name] = sl

        d = {
            'type': 'leaf',
            'meta': vars(sl.meta),
            'uid': self.uid + '_' + sl.name,
            'wmsName': sl.name,
            'clientOptions': {
                'visible': sl.is_visible,
            },
            'opacity': sl.opacity,
        }

        if sl.scale_range:
            d['zoom'] = {
                'minScale': sl.scale_range[0],
                'maxScale': sl.scale_range[1],
            }

        # @TODO get and reproject the source extent
        return d

    def mapproxy_config(self, mc, options=None):
        # we don't need to specify any layers for this config
        # but MP requires something for 'layers', so give it a dummy
        source = self.source.mapproxy_config(mc, {'layer_names': ['-']})

        # if there's no cache for this layer, don't make a grid for it
        if not self.cache.enabled:
            return mc.layer(self, {
                'title': self.uid,
                'sources': [source]
            })

        return super().mapproxy_config(mc, gws.defaults(options, source=source))

    @property
    def props(self):
        return gws.extend(super().props, {
            'url': gws.SERVER_ENDPOINT + '?cmd=mapHttpGetBbox&layerUid=' + self.uid,
            'layers': self.layers,
        })

    def description(self, options=None):
        return super().description(gws.defaults(
            options,
            sub_layers=[la for la in self.uid2leaf.values()]
        ))

    def render_bbox(self, bbox, width, height, **client_params):
        forward = {}

        if 'dpi' in client_params:
            forward['DPI__gws'] = client_params['dpi']

        if 'layers' in client_params:
            wms_names = []
            for uid in gws.as_list(client_params['layers']):
                la = self.uid2leaf.get(uid)
                if la:
                    wms_names.append(la.wms_name)

            forward['LAYERS__gws'] = ','.join(wms_names)

        # MPX only uses "layers" from the config, we need them to be dynamic
        # so it only works with local QGIS which 'knows' (thanks to our rewrite rules, see server)
        # how to rewrite 'LAYERS__gws' to 'LAYERS'

        return mpx.wms_request(
            self.uid,
            bbox,
            width,
            height,
            self.crs,
            forward)
