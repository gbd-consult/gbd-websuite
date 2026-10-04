"""Child layer configurations built from a hierarchy of source layers."""

from typing import Optional, cast
from collections.abc import Callable

import gws
import gws.gis.source
import gws.config.parser

from . import core


class FlattenConfig(gws.Config):
    """Turns source groups below a level into single layers."""

    level: int
    """Source hierarchy level from which groups become single layers."""
    useGroups: bool = False
    """Request a flattened group by its own name instead of its image layers."""


class Config(gws.Config):
    """Configuration for the layer tree."""

    rootLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to use as roots."""
    excludeLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to exclude."""
    flattenLayers: Optional[FlattenConfig]
    """Collapse source groups below a level into single layers."""
    autoLayers: Optional[list[core.AutoLayersConfig]]
    """Extra configuration for generated layers that match a filter."""


class _TreeConfigArgs(gws.Data):
    """Arguments for building a layer config tree."""

    root: gws.Root
    """Root object, provides the specs for parsing the generated configs."""
    source_layers: list[gws.SourceLayer]
    """Source layers to build the tree from."""
    roots_slf: gws.gis.source.LayerFilter
    """Filter for the source layers to use as roots."""
    exclude_slf: gws.gis.source.LayerFilter
    """Filter for the source layers to exclude."""
    flatten_config: FlattenConfig
    """Flattening options."""
    auto_layers: list[core.AutoLayersConfig]
    """Extra configurations merged into matching layers."""
    create_leaf_layer_config: Callable
    """Function that creates a leaf layer config from a list of source layers."""


def configure_group_layers_for(layer: core.Object, source_layers: list[gws.SourceLayer], create_leaf_layer_config: Callable) -> bool:
    """Create child layers of a layer from a list of source layers.

    The layer's ``rootLayers``, ``excludeLayers``, ``flattenLayers`` and
    ``autoLayers`` options control which source layers are used and how. Source
    groups become ``group`` layers, other source layers are passed to
    ``create_leaf_layer_config``.

    Args:
        layer: The layer to create children for.
        source_layers: Source layer hierarchy.
        create_leaf_layer_config: Function that takes a list of source layers
            and returns a config for a leaf layer that shows them.

    Returns:
        ``True``.

    Raises:
        ``gws.ConfigurationError``: If a generated config is invalid.
        ``gws.Error``: If no child layers could be created.
    """

    configs = _layer_configs_from_layer(layer, source_layers, create_leaf_layer_config)
    create_group_layers(layer, configs)
    return True


def create_group_layers(layer: core.Object, layer_configs: list):
    """Create child layers of a layer from their configs.

    Each config gets the parent's WGS extent, resolutions and map CRS. The
    created layers replace ``layer.layers``.

    Args:
        layer: The layer to create children for.
        layer_configs: Child layer configs.

    Raises:
        ``gws.Error``: If no child layers could be created.
    """

    layer.layers = []

    for cfg in layer_configs:
        cfg = gws.u.merge(
            cfg,
            _parentWgsExtent=layer.wgsExtent,
            _mapCrs=layer.mapCrs,
            _parentResolutions=layer.resolutions,
        )
        la = cast(gws.Layer, layer.create_child(gws.ext.object.layer, cfg))
        if la:
            layer.layers.append(la)

    if not layer.layers:
        raise gws.Error(f'group is empty: {layer}')


def _layer_configs_from_layer(layer: core.Object, source_layers: list[gws.SourceLayer], create_leaf_layer_config: Callable) -> list[gws.Config]:
    """Generate a config tree from a list of source layers and the main layer config."""

    return _layer_configs_from_args(
        _TreeConfigArgs(
            root=layer.root,
            source_layers=source_layers,
            roots_slf=layer.cfg('rootLayers'),
            exclude_slf=layer.cfg('excludeLayers'),
            flatten_config=layer.cfg('flattenLayers'),
            auto_layers=layer.cfg('autoLayers', default=[]),
            create_leaf_layer_config=create_leaf_layer_config,
        )
    )


def _layer_configs_from_args(tca: _TreeConfigArgs) -> list[gws.Config]:
    """Generate a config tree from a list of source layers."""

    # by default, take top-level layers as roots
    roots_slf = tca.roots_slf or gws.gis.source.LayerFilter(level=1)
    roots = gws.gis.source.filter_layers(tca.source_layers, roots_slf)

    # make configs...
    configs = gws.u.compact(_config(tca, sl, 0) for sl in roots)

    # configs need to be reparsed so that defaults can be injected
    layer_configs = []
    ctx = gws.ConfigContext(
        specs=tca.root.specs,
        readOptions={gws.SpecReadOption.acceptExtraProps, gws.SpecReadOption.allowMissing},
    )

    for c in configs:
        cfg = gws.config.parser.parse_dict(
            gws.u.to_dict(c),
            path='',
            as_type='gws.ext.config.layer',
            ctx=ctx,
        )
        if cfg:
            layer_configs.append(cfg)

    if ctx.errors:
        raise gws.ConfigurationError(ctx.errors[0].message)

    return layer_configs


def _config(tca: _TreeConfigArgs, sl: gws.SourceLayer, depth: int):
    """Build the config for a source layer, with title, visibility, opacity and auto layer configs."""
    cfg = _base_config(tca, sl, depth)
    if not cfg:
        return None

    cfg = gws.u.merge(
        gws.u.to_dict(cfg),
        {
            'title': sl.title,
            'clientOptions': {
                'hidden': not sl.isVisible,
                'expanded': sl.isExpanded,
            },
            'opacity': sl.opacity or 1,
        },
    )

    for cc in tca.auto_layers:
        if gws.gis.source.layer_matches(sl, cc.applyTo):
            cfg = _deep_merge(cfg, cc.config)

    return gws.u.compact(cfg)


def _base_config(tca: _TreeConfigArgs, sl: gws.SourceLayer, depth: int):
    """Build a leaf, flattened or group config for a source layer, or None if it is excluded or empty."""
    # source layer excluded by the filter
    if tca.exclude_slf and gws.gis.source.layer_matches(sl, tca.exclude_slf):
        return None

    # leaf layer
    if not sl.isGroup:
        return tca.create_leaf_layer_config([sl])

    # flattened group layer
    # NB use the absolute level to compute flatness, could also use relative (=depth)
    if tca.flatten_config and sl.aLevel >= tca.flatten_config.level:
        if tca.flatten_config.useGroups:
            return tca.create_leaf_layer_config([sl])

        slf = gws.gis.source.LayerFilter(isImage=True)
        leaves = gws.gis.source.filter_layers([sl], slf)
        if not leaves:
            return None
        return tca.create_leaf_layer_config(leaves)

    # ordinary group layer
    layer_cfgs = gws.u.compact(_config(tca, sub, depth + 1) for sub in sl.layers)
    if not layer_cfgs:
        return None
    return {
        'type': 'group',
        'layers': layer_cfgs,
    }


def _deep_merge(x, y):
    """Merge two values recursively: dicts by key, lists by concatenation, otherwise y unless it is None."""
    if (gws.u.is_dict(x) or gws.u.is_data_object(x)) and (gws.u.is_dict(y) or gws.u.is_data_object(y)):
        xd = gws.u.to_dict(x)
        yd = gws.u.to_dict(y)
        d = {k: _deep_merge(xd.get(k), yd.get(k)) for k in xd.keys() | yd.keys()}
        return d if gws.u.is_dict(x) else type(x)(d)

    if gws.u.is_list(x) and gws.u.is_list(y):
        return gws.u.compact(x) + gws.u.compact(y)

    return y if y is not None else x
