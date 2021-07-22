import gws
import gws.types as t
import gws.base.layer.image
import gws.base.layer.group
import gws.base.layer.core
import gws.base.ows.provider
import gws.config.parser
import gws.lib.legend
import gws.lib.ows
import gws.lib.gis
import gws.lib.gis
import gws.lib.gis
import gws.lib.zoom
from . import provider


@gws.ext.Config('layer.wms')
class Config(gws.base.layer.image.Config, provider.Config):
    rootLayers: t.Optional[gws.lib.gis.LayerFilter]  #: source layers to use as roots
    excludeLayers: t.Optional[gws.lib.gis.LayerFilter]  #: source layers to exclude
    flattenLayers: t.Optional[gws.base.layer.core.FlattenConfig]  #: flatten the layer hierarchy
    layerConfig: t.Optional[t.List[gws.base.layer.core.CustomConfig]]  #: custom configurations for specific layers


@gws.ext.Object('layer.wms')
class Object(gws.base.layer.group.BaseGroup):
    provider: provider.Object

    def configure(self):
        self.provider = gws.base.ows.provider.shared_object(self.root, provider.Object, self.config)

        if not self.has_configured.metadata:
            self.configure_metadata_from(self.provider.metadata)

        # by default, take the top-level layers as groups
        roots = gws.lib.gis.filter_layers(
            self.provider.source_layers,
            self.var('rootLayers') or gws.lib.gis.LayerFilter(level=1))

        layer_cfgs = []

        for sl in roots:
            cfg = self._layer(
                sl,
                exclude=self.var('excludeLayers'),
                flatten=self.var('flattenLayers'),
                custom_configs=self.var('layerConfig', default=[]),
                depth=1)
            if cfg:
                layer_cfgs.append(cfg)

        if gws.is_empty(layer_cfgs):
            raise gws.Error(f'no source layers in {self.uid!r}')

        group_cfg = {
            'type': 'group',
            'title': '',
            'layers': layer_cfgs
        }

        parsed_group_cfg = gws.config.parser.parse(self.root.specs, group_cfg, 'gws.ext.layer.group.Config')
        self.configure_layers_from(parsed_group_cfg.layers)

    def _layer(
            self,
            sl: gws.lib.gis.SourceLayer,
            exclude: gws.lib.gis.LayerFilter,
            flatten: gws.base.layer.core.FlattenConfig,
            custom_configs: t.List[gws.base.layer.core.CustomConfig],
            depth: int
    ):

        if exclude and gws.lib.gis.layer_matches(sl, exclude):
            return

        la = None

        if not sl.is_group:
            # ordinary image layer
            la = self._image_layer([sl.name])

        elif flatten and sl.a_level >= flatten.level:
            # flattened group layer
            # NB use the absolute level to compute flatness, could also use relative (=depth)
            if self.flatten.useGroups:
                la = self._image_layer([sl.name])
            else:
                ls = gws.lib.gis.image_layers(sl)
                if ls:
                    la = self._image_layer([s.name for s in ls])
        else:
            # ordinary group layer
            layers = gws.compact(self._layer(la, exclude, flatten, custom_configs, depth + 1) for la in sl.layers)
            if layers:
                la = {
                    'type': 'group',
                    'uid': gws.as_uid(sl.name),
                    'layers': layers
                }

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

        custom = [gws.strip(c) for c in custom_configs if gws.lib.gis.layer_matches(sl, c.applyTo)]
        if custom:
            la = gws.deep_merge(la, *custom)
            la.pop('applyTo', None)

        return gws.compact(la)

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
        cfg['type'] = 'wmsflat'
        cfg['sourceLayers'] = {'names': source_layer_names}
        return cfg
