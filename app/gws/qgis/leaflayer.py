import gws.common.layer
import gws.gis.shape
import gws.gis.feature
import gws.gis.proj
import gws.common.db
import gws.tools.json2

import gws.types as t


class Config(gws.common.layer.Config):
    """QGIS Leaf layer"""

    pass


class Object(gws.common.layer.Layer):
    def configure(self):
        super().configure()
        self.layer = self.var('_layer')
        self.source_layers = self.var('_source_layers')

    def configure_legend(self):
        return super().configure_legend() or t.LayerLegend(enabled=True)

    def render_legend_image(self, context=None):
        return self.layer.provider.get_legend(self.source_layers, self.legend.options)

    @property
    def props(self):
        p = super().props

        return gws.merge(p, {
            'type': 'leaf',
        })
