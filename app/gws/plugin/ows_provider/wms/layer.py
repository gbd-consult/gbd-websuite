import gws
import gws.base.layer
import gws.lib.gis
import gws.types as t

from . import provider as provider_module


@gws.ext.Config('layer.wms')
class Config(gws.base.layer.image.Config, provider_module.Config):
    rootLayers: t.Optional[gws.lib.gis.SourceLayerFilter]  #: source layers to use as roots
    excludeLayers: t.Optional[gws.lib.gis.SourceLayerFilter]  #: source layers to exclude
    flattenLayers: t.Optional[gws.base.layer.types.FlattenConfig]  #: flatten the layer hierarchy
    layerConfig: t.Optional[t.List[gws.base.layer.types.CustomConfig]]  #: custom configurations for specific layers


@gws.ext.Object('layer.wms')
class Object(gws.base.layer.group.BaseGroup):
    provider: provider_module.Object

    def configure(self):
        pass

    def configure_source(self):
        self.provider = self.root.create_object(provider_module.Object, self.config, shared=True)

        cfgs = self.layer_tree_configuration(
            source_layers=self.provider.source_layers,
            roots_slf=self.var('rootLayers'),
            exclude_slf=self.var('excludeLayers'),
            flatten=self.var('flattenLayers'),
            custom_configs=self.var('layerConfig'),
            leaf_config=self.leaf_config
        )

        if not cfgs:
            raise gws.ConfigurationError(f'no source layers in {self.uid!r}')

        self.configure_layers(cfgs)
        return True

    def configure_metadata(self):
        if not super().configure_metadata():
            self.set_metadata(self.provider.metadata)
            return True

    def leaf_config(self, source_layers):
        return {
            'type': 'wmsflat',
            '_provider': self.provider,
            '_source_layers': source_layers,
        }
