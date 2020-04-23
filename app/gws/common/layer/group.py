"""Generic group layer."""

import gws
import gws.gis.legend

import gws.types as t

from . import layer


class Group(layer.Layer):
    def configure(self):
        super().configure()

        self.is_group = True

        self.supports_wms = True
        self.supports_wfs = True

        self.layers: t.List[t.ILayer] = []

    def configure_legend(self):
        # since the sub layers are (post) configured after us, there's no way to tell
        # if any sublayer actually has a legend. So assume we have a legend for now
        return super().configure_legend() or t.LayerLegend(enabled=True)

    def ows_enabled(self, service):
        return super().ows_enabled(service) and any(la.ows_enabled(service) for la in self.layers)

    def render_legend_image(self, context=None):
        paths = gws.compact(la.render_legend(context) for la in self.layers if la.has_legend)
        return gws.gis.legend.combine_legend_paths(paths)

    @property
    def props(self):
        return gws.merge(super().props, type='group', layers=self.layers)