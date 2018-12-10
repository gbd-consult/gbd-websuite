import gws
import gws.types as t
import gws.gis.layer
import gws.gis.source


class Config(gws.gis.layer.ProxiedConfig):
    """box layer"""
    pass


class LayerProps(gws.gis.layer.BaseProps):
    url: str


class Object(gws.gis.layer.Proxied):
    def __init__(self):
        super().__init__()
        self.source_layers: t.List[t.SourceLayer] = []
        self.source_layer_names: t.List[str] = []

    def configure(self):
        super().configure()

        # filter source layers
        slf = self.var('sourceLayers')
        ls = [la for la in gws.gis.source.filter_layers(self.source.layers, slf)]

        # additonaly to the filter, ensure only image layers
        ls = [la for la in ls if la.is_image]

        if not ls:
            raise ValueError(f'no layers found in {self.uid!r}')

        self.source_layers = ls
        self.source_layer_names = gws.compact(la.name for la in ls)

    def mapproxy_config(self, mc, options=None):
        source = self.source.mapproxy_config(mc, {'layer_names': self.source_layer_names})

        # if there's no cache for this layer, don't make a grid for it
        if not self.cache.enabled:
            return mc.layer(self, {
                'title': self.uid,
                'sources': [source]
            })

        return super().mapproxy_config(mc, gws.defaults(options, source=source))

    def description(self, options=None):
        sub_layers = self.source_layers
        if len(sub_layers) == 1 and gws.get(sub_layers[0], 'title') == self.title:
            sub_layers = []

        return super().description(gws.defaults(
            options,
            sub_layers=sub_layers))

    @property
    def props(self):
        return gws.extend(super().props, {
            'url': gws.SERVER_ENDPOINT + '?cmd=mapHttpGetBbox&layerUid=' + self.uid,
        })

