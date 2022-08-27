"""Generic group layer."""

import gws
import gws.config
import gws.gis.source
import gws.types as t

from . import lib, main


@gws.ext.config.layer('group')
class Config(main.Config):
    """Group layer"""

    layers: t.List[gws.ext.config.layer]  #: layers in this group


class FlattenOption(gws.Config):
    """Layer hierarchy flattening"""

    level: int  #: flatten level
    useGroups: bool = False  #: use group names (true) or image layer names (false)


class TreeConfig(main.Config):
    rootLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use as roots
    excludeLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to exclude
    flattenLayers: t.Optional[FlattenOption]  #: flatten the layer hierarchy
    layerConfig: t.Optional[t.List[main.CustomConfig]]  #: custom configurations for specific layers


@gws.ext.object.layer('group')
class Object(main.Object):
    def props(self, user):
        return gws.merge(super().props(user), type='group')

    def configure(self):
        super().configure()

        self.layers = []
        for cfg in self.var('layers', default=[]):
            cfg.defaultCrs = self.crs
            cfg.defaultResolutions = self.var('defaultResolutions')
            self.layers.append(self.create_child(gws.ext.object.layer, cfg))

        # self.supports_raster_ows = any(la.supports_raster_ows for la in self.layers)
        # self.supports_vector_ows = any(la.supports_vector_ows for la in self.layers)

    def configure_layer_tree(self, source_layers: t.List[gws.SourceLayer], leaf_layer_config_fn: t.Callable):
        cfgs = _layer_tree_configuration(
            source_layers=source_layers,
            roots_slf=self.var('rootLayers'),
            exclude_slf=self.var('excludeLayers'),
            flatten=self.var('flattenLayers'),
            custom_configs=self.var('layerConfig', default=[]),
            leaf_layer_config_fn=leaf_layer_config_fn
        )
        if not cfgs:
            raise gws.ConfigurationError(f'no source layers in {self.uid!r}')

        # configs need to be reparsed so that defaults can be injected
        cfgs = [
            gws.config.parse(self.root.specs, c, 'gws.ext.config.layer', with_internal_objects=True)
            for c in cfgs
        ]

        self.configure_layers(cfgs)

    def configure_legend(self):
        if not super().configure_legend():
            legend_layers = [la for la in self.layers if la.has_legend]
            if legend_layers:
                self.legend = gws.Legend(
                    enabled=True,
                    layers=legend_layers,
                    options=self.var('legend.options', default={}))
                return True

    def configure_zoom(self):
        if not super().configure_zoom():
            resolutions = set()
            for la in self.layers:
                resolutions.update(la.resolutions)
            self.resolutions = sorted(resolutions)
            return True


##

def _layer_tree_configuration(
        source_layers: t.List[gws.SourceLayer],
        roots_slf: gws.gis.source.LayerFilter,
        exclude_slf: gws.gis.source.LayerFilter,
        flatten: FlattenOption,
        custom_configs: t.List[main.CustomConfig],
        leaf_layer_config_fn: t.Callable,
) -> t.List[gws.Config]:
    ##

    def walk(sl: gws.SourceLayer, depth: int):
        if exclude_slf and gws.gis.source.layer_matches(sl, exclude_slf):
            return None

        cfg = None

        if not sl.is_group:
            # leaf layer
            cfg = leaf_layer_config_fn([sl])

        elif flatten and sl.a_level >= flatten.level:
            # flattened group layer
            # NB use the absolute level to compute flatness, could also use relative (=depth)
            if flatten.useGroups:
                cfg = leaf_layer_config_fn([sl])
            else:
                slf = gws.gis.source.LayerFilter(is_image=True)
                image_layers = gws.gis.source.filter_layers([sl], slf)
                if not image_layers:
                    return None
                cfg = leaf_layer_config_fn(image_layers)

        else:
            # ordinary group layer
            sub_cfgs = gws.compact(walk(sub, depth + 1) for sub in sl.layers)
            if not sub_cfgs:
                return None
            cfg = {
                'type': 'group',
                'uid': gws.to_uid(sl.name),
                'layers': sub_cfgs
            }

        if not cfg:
            return

        cfg = gws.merge(cfg, {
            'uid': gws.to_uid(sl.name),
            'title': sl.title,
            'clientOptions': {
                'visible': sl.is_visible,
                'expanded': sl.is_expanded,
            },
            'opacity': sl.opacity or 1,
        })

        if sl.scale_range:
            cfg['zoom'] = {
                'minScale': sl.scale_range[0],
                'maxScale': sl.scale_range[1],
            }

        for flt, cc in zip(custom_filters, custom_configs):
            if gws.gis.source.layer_matches(sl, flt):
                cfg = gws.deep_merge(vars(cc), cfg)
            cfg.pop('applyTo', None)

        return gws.compact(cfg)

    ##

    custom_filters = [
        gws.gis.source.layer_filter_from_config(cc.applyTo)
        for cc in custom_configs
    ]

    # by default, take top-level layers as roots

    roots = gws.gis.source.filter_layers(
        source_layers,
        roots_slf or gws.gis.source.LayerFilter(level=1))

    # make configs...

    return gws.compact(walk(sl, 0) for sl in roots)
