import gws
import gws.base.layer
import gws.base.ows
import gws.lib.gis
import gws.lib.legend
import gws.lib.ows
import gws.lib.zoom
import gws.types as t
from . import provider


@gws.ext.Config('layer.wms')
class Config(gws.base.layer.image.Config, provider.Config):
    rootLayers: t.Optional[gws.lib.gis.SourceLayerFilter]  #: source layers to use as roots
    excludeLayers: t.Optional[gws.lib.gis.SourceLayerFilter]  #: source layers to exclude
    flattenLayers: t.Optional[gws.base.layer.types.FlattenConfig]  #: flatten the layer hierarchy
    layerConfig: t.Optional[t.List[gws.base.layer.types.CustomConfig]]  #: custom configurations for specific layers


@gws.ext.Object('layer.wms')
class Object(gws.base.layer.group.BaseGroup):
    provider: provider.Object

    def configure(self):
        self.provider = provider.create(self.root, self.config, shared=True)

        if not self.has_configured_metadata:
            self.configure_metadata_from(self.provider.metadata)

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

    def leaf_config(self, source_layers):
        return {
            'type': 'wmsflat',
            '_provider': self.provider,
            '_source_layers': source_layers,
        }
