import gws.gis.layer
import gws.types as t


class Config(gws.gis.layer.BaseConfig):
    """leaf layer"""

    wmsName: str #: this layer ID in the WMS service


class LayerProps(gws.gis.layer.BaseProps):
    pass


class Object(gws.gis.layer.Base):
    def __init__(self):
        super().__init__()
        self.wms_name = None
        self.tree_layer: t.LayerObject = None
        self.source_layer = t.SourceLayer

    def configure(self):
        super().configure()
        self.wms_name = self.var('wmsName')
        # NB 'tree_layer' and 'source_layer' are set by the owner

    def description(self, options=None):
        return super().description(gws.defaults(options, source_layers=[self.source_layer]))

    def render_bbox(self, bbox, width, height, **client_params):
        # to render just this layer (for printing), ask the master to render it
        # @TODO: actually, the printer should collect leaf layers

        client_params['layers'] = self.uid
        return self.tree_layer.render_bbox(bbox, width, height, **client_params)
