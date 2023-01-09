"""Structures and utilities for tree layers."""

import gws
import gws.gis.source
import gws.config.parser
import gws.types as t

from . import core


class FlattenConfig(gws.Config):
    """Layer hierarchy flattening"""

    level: int 
    """flatten level"""
    useGroups: bool = False 
    """use group names (true) or image layer names (false)"""


class Config(gws.Config):
    rootLayers: t.Optional[gws.gis.source.LayerFilter] 
    """source layers to use as roots"""
    excludeLayers: t.Optional[gws.gis.source.LayerFilter] 
    """source layers to exclude"""
    flattenLayers: t.Optional[FlattenConfig] 
    """flatten the layer hierarchy"""
    layerConfig: t.Optional[t.List[core.CustomConfig]] 
    """custom configurations for specific layers"""


def layer_configs_from_layer(
        layer: core.Object,
        source_layers: t.List[gws.SourceLayer],
        leaf_layer_maker: t.Callable
) -> t.List[gws.Config]:
    ##
    return layer_configs_from_args(
        layer.root,
        source_layers=source_layers,
        roots_slf=layer.var('rootLayers'),
        exclude_slf=layer.var('excludeLayers'),
        flatten_config=layer.var('flattenLayers'),
        custom_configs=layer.var('layerConfig', default=[]),
        leaf_layer_maker=leaf_layer_maker
    )


def layer_configs_from_args(
        root: gws.IRoot,
        source_layers: t.List[gws.SourceLayer],
        roots_slf: gws.gis.source.LayerFilter,
        exclude_slf: gws.gis.source.LayerFilter,
        flatten_config: FlattenConfig,
        custom_configs: t.List[core.CustomConfig],
        leaf_layer_maker: t.Callable,
) -> t.List[gws.Config]:
    ##

    def base_config(sl: gws.SourceLayer, depth: int):
        if exclude_slf and gws.gis.source.layer_matches(sl, exclude_slf):
            return None

        if not sl.isGroup:
            # leaf layer
            return leaf_layer_maker([sl])

        if flatten_config and sl.aLevel >= flatten_config.level:
            # flattened group layer
            # NB use the absolute level to compute flatness, could also use relative (=depth)

            if flatten_config.useGroups:
                return leaf_layer_maker([sl])

            slf = gws.gis.source.LayerFilter(isImage=True)
            leaves = gws.gis.source.filter_layers([sl], slf)
            if not leaves:
                return None

            return leaf_layer_maker(leaves)

        # ordinary group layer

        layer_cfgs = gws.compact(walk(sub, depth + 1) for sub in sl.layers)
        if not layer_cfgs:
            return None

        return {
            'type': 'group',
            'layers': layer_cfgs,
        }

    def walk(sl: gws.SourceLayer, depth: int):
        cfg = base_config(sl, depth)
        if not cfg:
            return None

        cfg = gws.merge(cfg, {
            'title': sl.title,
            'clientOptions': {
                'visible': sl.isVisible,
                'expanded': sl.isExpanded,
            },
            'opacity': sl.opacity or 1,
        })

        for flt, custom in zip(custom_filters, custom_configs):
            if gws.gis.source.layer_matches(sl, flt):
                cfg = gws.merge(custom, cfg)
                cfg.pop('applyTo', None)

        return gws.compact(cfg)

    ##

    custom_filters = [
        cc.applyTo
        for cc in custom_configs
    ]

    # by default, take top-level layers as roots

    roots_slf = roots_slf or gws.gis.source.LayerFilter(level=1)
    roots = gws.gis.source.filter_layers(source_layers, roots_slf)

    # make configs...

    configs = gws.compact(walk(sl, 0) for sl in roots)

    # configs need to be reparsed so that defaults can be injected

    return [
        gws.config.parser.parse(
            root.specs,
            cfg,
            'gws.ext.config.layer',
            read_options={'accept_extra_props', 'relax_required'}
        )
        for cfg in configs
    ]
