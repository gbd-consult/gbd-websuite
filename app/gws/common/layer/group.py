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


    def configure_legend(self):
        # since the sub layers are (post) configured after us, there's no way to tell
        # if any sublayer actually has a legend. So assume we have a legend for now
        return super().configure_legend() or t.LayerLegend(enabled=True)

    def render_legend_image(self, context=None):
        paths = gws.compact(la.render_legend(context) for la in self.layers if la.has_legend)
        return gws.gis.legend.combine_legend_paths(paths)

    @property
    def props(self):
        resolutions = set()
        for la in self.layers:
            resolutions.update(la.resolutions)
        return gws.merge(
            super().props,
            type='group',
            layers=self.layers,
            resolutions=sorted(resolutions, reverse=True))
