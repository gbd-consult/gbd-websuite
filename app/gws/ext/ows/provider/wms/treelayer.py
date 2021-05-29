import gws
import gws.config.parser
import gws.common.layer
import gws.gis.legend
import gws.gis.ows
import gws.gis.source
import gws.gis.source
import gws.gis.util
import gws.gis.zoom

import gws.types as t

from . import provider


class Config(gws.common.layer.ImageConfig, provider.Config):
    rootLayers: t.Optional[gws.gis.source.LayerFilter]  #: source layers to use as roots
    excludeLayers: t.Optional[gws.gis.source.LayerFilter]  #: source layers to exclude
    flattenLayers: t.Optional[gws.common.layer.types.FlattenConfig]  #: flatten the layer hierarchy
    layerConfig: t.Optional[t.List[gws.common.layer.CustomConfig]]  #: custom configurations for specific layers


class Object(gws.common.layer.Group):
    def configure(self):
        super().configure()

        self.provider = gws.gis.ows.shared_provider(provider.Object, self, self.config)

        self.meta = self.configure_metadata(self.provider.meta)
        self.title = self.meta.title

        # similar to qgis/layer

        # by default, take the top-level layers as groups
        slf = self.var('rootLayers') or gws.gis.source.LayerFilter(level=1)
        self.root_layers = gws.gis.source.filter_layers(self.provider.source_layers, slf)
        self.exclude_layers = self.var('excludeLayers')
        self.flatten = self.var('flattenLayers')
        self.custom_layer_config = self.var('layerConfig', default=[])

        layer_cfgs = gws.compact(self._layer(sl, depth=1) for sl in self.root_layers)
        if gws.is_empty(layer_cfgs):
            raise gws.Error(f'no source layers in {self.uid!r}')

        top_group = {
            'type': 'group',
            'title': '',
            'layers': layer_cfgs
        }

        top_cfg = gws.config.parser.parse(top_group, 'gws.ext.layer.group.Config')
        self.layers = gws.common.layer.add_layers_to_object(self, top_cfg.layers)

    def _layer(self, sl: t.SourceLayer, depth: int):
        if self.exclude_layers and gws.gis.source.layer_matches(sl, self.exclude_layers):
            return

        if sl.is_group:
            # NB use the absolute level to compute flatness, could also use relative (=depth)
            if self.flatten and sl.a_level >= self.flatten.level:
                la = self._flat_group_layer(sl)
            else:
                la = self._group_layer(sl, depth)
        else:
            la = self._image_layer([sl.name])

        if not la:
            return

        la = gws.merge(la, {
            'uid': gws.as_uid(sl.name),
            'title': sl.title,
            'clientOptions': {
                'visible': sl.is_visible,
            },
            'opacity': sl.opacity or 1,
        })

        if sl.scale_range:
            la['zoom'] = {
                'minScale': sl.scale_range[0],
                'maxScale': sl.scale_range[1],
            }

        custom = [gws.strip(c) for c in self.custom_layer_config if gws.gis.source.layer_matches(sl, c.applyTo)]
        if custom:
            la = gws.deep_merge(la, *custom)
            if la.applyTo:
                delattr(la, 'applyTo')

        return gws.compact(la)

    def _group_layer(self, sl: t.SourceLayer, depth: int):
        layers = gws.compact(self._layer(la, depth + 1) for la in sl.layers)
        if not layers:
            return
        return {
            'type': 'group',
            'title': sl.title,
            'uid': gws.as_uid(sl.name),
            'layers': layers
        }

    def _flat_group_layer(self, sl: t.SourceLayer):
        if self.flatten.useGroups:
            names = [sl.name]
        else:
            ls = gws.gis.source.image_layers(sl)
            if not ls:
                return
            names = [s.name for s in ls]

        return self._image_layer(names)

    _copy_keys = (
        'capsLayersBottomUp',
        'getCapabilitiesParams',
        'getMapParams',
        'invertAxis',
        'maxRequests',
        'url',
    )

    def _image_layer(self, source_layer_names):
        cfg = gws.compact({k: self.config.get(k) for k in self._copy_keys})
        cfg['type'] = 'wms'
        cfg['sourceLayers'] = {'names': source_layer_names}
        return cfg
