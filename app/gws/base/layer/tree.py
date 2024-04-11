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
    autoLayers: t.Optional[list[core.AutoLayersOptions]]
    """custom configurations for automatically created layers"""


class TreeConfigArgs(gws.Data):
    root: gws.IRoot
    source_layers: list[gws.SourceLayer]
    roots_slf: gws.gis.source.LayerFilter
    exclude_slf: gws.gis.source.LayerFilter
    flatten_config: FlattenConfig
    auto_layers: list[core.AutoLayersOptions]
    leaf_layer_maker: t.Callable


def layer_configs_from_layer(layer: core.Object, source_layers: list[gws.SourceLayer], leaf_layer_maker: t.Callable) -> list[gws.Config]:
    """Generate a config tree from a list of source layers and the main layer config."""

    return layer_configs_from_args(TreeConfigArgs(
        root=layer.root,
        source_layers=source_layers,
        roots_slf=layer.cfg('rootLayers'),
        exclude_slf=layer.cfg('excludeLayers'),
        flatten_config=layer.cfg('flattenLayers'),
        auto_layers=layer.cfg('autoLayers', default=[]),
        leaf_layer_maker=leaf_layer_maker
    ))


def layer_configs_from_args(tca: TreeConfigArgs) -> list[gws.Config]:
    """Generate a config tree from a list of source layers."""

    # by default, take top-level layers as roots
    roots_slf = tca.roots_slf or gws.gis.source.LayerFilter(level=1)
    roots = gws.gis.source.filter_layers(tca.source_layers, roots_slf)

    # make configs...
    configs = gws.compact(_config(tca, sl, 0) for sl in roots)

    # configs need to be reparsed so that defaults can be injected
    return [
        gws.config.parser.parse(
            tca.root.specs,
            cfg,
            'gws.ext.config.layer',
            read_options={gws.SpecReadOption.acceptExtraProps, gws.SpecReadOption.allowMissing}
        )
        for cfg in configs
    ]


def _config(tca: TreeConfigArgs, sl: gws.SourceLayer, depth: int):
    cfg = _base_config(tca, sl, depth)
    if not cfg:
        return None

    cfg = gws.merge(gws.to_dict(cfg), {
        'title': sl.title,
        'clientOptions': {
            'hidden': not sl.isVisible,
            'expanded': sl.isExpanded,
        },
        'opacity': sl.opacity or 1,
    })

    for cc in tca.auto_layers:
        if gws.gis.source.layer_matches(sl, cc.applyTo):
            cfg = gws.deep_merge(cc.config, cfg)

    return gws.compact(cfg)


def _base_config(tca: TreeConfigArgs, sl: gws.SourceLayer, depth: int):
    # source layer excluded by the filter
    if tca.exclude_slf and gws.gis.source.layer_matches(sl, tca.exclude_slf):
        return None

    # leaf layer
    if not sl.isGroup:
        return tca.leaf_layer_maker([sl])

    # flattened group layer
    # NB use the absolute level to compute flatness, could also use relative (=depth)
    if tca.flatten_config and sl.aLevel >= tca.flatten_config.level:

        if tca.flatten_config.useGroups:
            return tca.leaf_layer_maker([sl])

        slf = gws.gis.source.LayerFilter(isImage=True)
        leaves = gws.gis.source.filter_layers([sl], slf)
        if not leaves:
            return None
        return tca.leaf_layer_maker(leaves)

    # ordinary group layer
    layer_cfgs = gws.compact(_config(tca, sub, depth + 1) for sub in sl.layers)
    if not layer_cfgs:
        return None
    return {
        'type': 'group',
        'layers': layer_cfgs,
    }
