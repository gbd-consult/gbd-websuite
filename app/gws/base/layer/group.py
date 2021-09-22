"""Generic group layer."""

import gws
import gws.config
import gws.lib.gis
import gws.lib.legend
import gws.types as t
from . import core, types


class BaseGroup(core.Object):
    is_group = True

    @property
    def props(self):
        return gws.merge(
            super().props,
            type='group',
            layers=self.layers,
        )

    def configure_layers(self, cfgs: t.List[gws.Config]):
        self.layers = t.cast(t.List[gws.ILayer], self.create_children('gws.ext.layer', cfgs))

        if not self.has_configured_legend:
            self.legend = gws.Legend(enabled=any(la.has_legend for la in self.layers))
            self.has_configured_legend = True

        if not self.has_configured_resolutions:
            resolutions = set()
            for la in self.layers:
                resolutions.update(la.resolutions)
            self.resolutions = sorted(resolutions)
            self.has_configured_resolutions = True

        self.supports_wms = any(la.supports_wms for la in self.layers)
        self.supports_wfs = any(la.supports_wfs for la in self.layers)

        self.has_configured_layers = True

    def render_legend(self, context=None):
        sup = super().render_legend(context)
        if sup:
            return sup
        return gws.lib.legend.combine_outputs([la.get_legend(context) for la in self.layers], self.legend.options)

    def layer_tree_configuration(
            self,
            source_layers: t.List[gws.lib.gis.SourceLayer],
            roots_slf: gws.lib.gis.SourceLayerFilter,
            exclude_slf: gws.lib.gis.SourceLayerFilter,
            flatten: types.FlattenConfig,
            custom_configs: t.List[types.CustomConfig],
            layer_config_factory: t.Callable[[t.List[gws.lib.gis.SourceLayer]], dict]
    ):
        def _make_config(sl, depth):
            cfg = _base_config(sl, depth)
            if not cfg:
                return

            cfg = gws.merge(cfg, {
                'uid': gws.as_uid(sl.name),
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

            if custom_configs:
                custom = [cc for cc in custom_configs if gws.lib.gis.source_layer_matches(sl, cc.applyTo)]
                if custom:
                    cfg = gws.deep_merge(cfg, *custom)
                    cfg.pop('applyTo', None)

            return gws.compact(cfg)

        def _base_config(sl, depth):
            if exclude_slf and gws.lib.gis.source_layer_matches(sl, exclude_slf):
                return None

            if not sl.is_group:
                # leaf layer
                return layer_config_factory([sl])

            if flatten and sl.a_level >= flatten.level:
                # flattened group layer
                # NB use the absolute level to compute flatness, could also use relative (=depth)
                if flatten.useGroups:
                    return layer_config_factory([sl])
                image_layers = gws.lib.gis.enum_source_layers([sl], is_image=True)
                if image_layers:
                    return layer_config_factory(image_layers)
                return None

            # ordinary group layer
            configs = gws.compact(_make_config(sub, depth + 1) for sub in sl.layers)
            if configs:
                return {
                    'type': 'group',
                    'uid': gws.as_uid(sl.name),
                    'layers': configs
                }

        # by default, take top-level layers as roots

        roots = gws.lib.gis.filter_source_layers(
            source_layers,
            roots_slf or gws.lib.gis.SourceLayerFilter(level=1))

        cfgs = []
        for sl in roots:
            cfg = _make_config(sl, 0)
            if cfg:
                cfgs.append(gws.config.parse(self.root.specs, cfg, 'gws.ext.layer.Config'))
        return cfgs


@gws.ext.Config('layer.group')
class Config(types.Config):
    """Group layer"""

    layers: t.List[gws.ext.layer.Config]  #: layers in this group


@gws.ext.Object('layer.group')
class Object(BaseGroup):

    def configure(self):
        self.configure_layers(self.var('layers'))
