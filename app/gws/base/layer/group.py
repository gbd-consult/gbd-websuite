"""Generic group layer."""

import gws
import gws.types as t
import gws.lib.legend

from . import core


class BaseGroup(core.Object):
    is_group = True

    @property
    def props(self):
        return gws.merge(
            super().props,
            type='group',
            layers=self.layers,
            resolutions=self.resolutions,
        )

    def configure_layers_from(self, configs: t.List[gws.Config]):
        self.layers = [self.create_child('gws.ext.layer', c) for c in configs]

        if not self.has_configured.legend:
            self.legend = gws.Legend(enabled=any(la.has_legend for la in self.layers))
            self.has_configured.legend = True

        if not self.has_configured.resolutions:
            resolutions = set()
            for la in self.layers:
                resolutions.update(la.resolutions)
            self.resolutions = sorted(resolutions)
            self.has_configured.resolutions = True

        self.supports_wms = any(la.supports_wms for la in self.layers)
        self.supports_wfs = any(la.supports_wfs for la in self.layers)

    def render_legend(self, context=None):
        sup = super().render_legend(context)
        if sup:
            return sup
        return gws.lib.legend.combine_outputs([la.get_legend(context) for la in self.layers], self.legend.options)


@gws.ext.Config('layer.group')
class Config(core.Config):
    """Group layer"""

    layers: t.List[gws.ext.layer.Config]  #: layers in this group


@gws.ext.Object('layer.group')
class Object(BaseGroup):

    def configure(self):
        self.configure_layers_from(self.var('layers'))

