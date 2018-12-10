import gws
import gws.gis.layer
import gws.types as t
import gws.gis.source


class Config(gws.gis.layer.ProxiedConfig):
    """tile layer"""
    pass


class LayerProps(gws.gis.layer.BaseProps):
    url: str
    tileSize: int

# @TODO merge with box

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

        self.source_layers = ls
        self.source_layer_names = gws.compact(la.name for la in ls)

        # force metaSize=1 for tiled layers, otherwise MP keeps requested the same tile multiple times
        self.grid = t.GridConfig(gws.extend(self.grid, {
            'metaSize': 1,
            'metaBuffer': 0,
        }))

    def mapproxy_config(self, mc, options=None):
        source = self.source.mapproxy_config(mc, {'layer_names': self.source_layer_names})
        return super().mapproxy_config(mc, gws.defaults(options, source=source))

    @property
    def props(self):
        return gws.extend(super().props, {
            'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetXyz/layerUid/' + self.uid + '/z/{z}/x/{x}/y/{y}/t.png',
            'tileSize': self.grid.tileSize,
            'extent': self.extent,

        })
