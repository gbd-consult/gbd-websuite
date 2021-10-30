"""Generic group layer."""

import gws
import gws.config
import gws.lib.gis
import gws.lib.legend
import gws.types as t
from . import core, types


class BaseGroup(core.Object):
    is_group = True

    def props_for(self, user):
        return gws.merge(super().props_for(user), type='group', layers=self.layers)

    def configure_layers(self, cfgs: t.List[gws.Config]):
        self.layers = self.create_children('gws.ext.layer', cfgs)
        self.supports_raster_ows = any(la.supports_raster_ows for la in self.layers)
        self.supports_vector_ows = any(la.supports_vector_ows for la in self.layers)

    def configure_legend(self):
        if not super().configure_legend():
            legend_layers = [la for la in self.layers if la.has_legend]
            if legend_layers:
                self.legend = gws.Legend(enabled=True, layers=legend_layers)
                return True

    def configure_zoom(self):
        if not super().configure_zoom():
            resolutions = set()
            for la in self.layers:
                resolutions.update(la.resolutions)
            self.resolutions = sorted(resolutions)
            return True

    def layer_tree_configuration(
            self,
            source_layers: t.List[gws.lib.gis.SourceLayer],
            roots_slf: gws.lib.gis.SourceLayerFilter,
            exclude_slf: gws.lib.gis.SourceLayerFilter,
            flatten: types.FlattenConfig,
            custom_configs: t.List[types.CustomConfig],
            leaf_config: t.Callable[[t.List[gws.lib.gis.SourceLayer]], dict]
    ):
        def _make_config(sl, depth):
            cfg = _base_config(sl, depth)
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

            if custom_configs:
                custom = [cc for cc in custom_configs if gws.lib.gis.source_layer_matches(sl, cc.applyTo)]
                if custom:
                    cfg = gws.deep_merge(*custom, cfg)
                    cfg.pop('applyTo', None)

            return gws.compact(cfg)

        def _base_config(sl, depth):
            if exclude_slf and gws.lib.gis.source_layer_matches(sl, exclude_slf):
                return None

            if not sl.is_group:
                # leaf layer
                return leaf_config([sl])

            if flatten and sl.a_level >= flatten.level:
                # flattened group layer
                # NB use the absolute level to compute flatness, could also use relative (=depth)
                if flatten.useGroups:
                    return leaf_config([sl])
                image_layers = gws.lib.gis.enum_source_layers([sl], is_image=True)
                if image_layers:
                    return leaf_config(image_layers)
                return None

            # ordinary group layer
            configs = gws.compact(_make_config(sub, depth + 1) for sub in sl.layers)
            if configs:
                return {
                    'type': 'group',
                    'uid': gws.to_uid(sl.name),
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
                cfgs.append(gws.config.parse(self.root.specs, cfg, 'gws.ext.layer.Config', with_internal_objects=True))
        return cfgs


@gws.ext.Config('layer.group')
class Config(types.Config):
    """Group layer"""

    layers: t.List[gws.ext.layer.Config]  #: layers in this group


@gws.ext.Object('layer.group')
class Object(BaseGroup):

    def configure(self):
        self.configure_layers(self.var('layers'))
