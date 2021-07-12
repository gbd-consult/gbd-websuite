"""Generic group layer."""

import gws
import gws.types as t
import gws.lib.legend

from . import core


@gws.ext.Config('layer.group')
class Config(core.Config):
    """Group layer"""

    layers: t.List[gws.ext.layer.Config]  #: layers in this group


@gws.ext.Object('layer.group')
class Object(core.Object):
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

    def configure(self):
        self.is_group = True
        self.layers = core.add_layers_to_object(self, self.var('layers'))
        self.supports_wms = any(la.supports_wms for la in self.layers)
        self.supports_wfs = any(la.supports_wfs for la in self.layers)

    def configure_legend(self):
        # since the sub layers are (post) configured after us, there's no way to tell
        # if any sublayer actually has a legend. So assume we have a legend for now
        return super().configure_legend() or gws.Legend(enabled=True)

    def render_legend_to_path(self, context=None):
        paths = gws.compact(la.render_legend_to_path(context) for la in self.layers if la.has_legend)
        return gws.lib.legend.combine_legend_paths(paths)
