"""QGIS legend."""

import gws
import gws.base.legend
import gws.config.util
import gws.lib.mime
import gws.gis.source
import gws.lib.image

import gws.types as t

from . import provider

gws.ext.new.legend('qgis')

# see https://docs.qgis.org/3.22/de/docs/server_manual/services/wms.html#getlegendgraphics

_DEFAULT_LEGEND_PARAMS = {
    'BOXSPACE': 2,
    'ICONLABELSPACE': 2,
    'ITEMFONTBOLD': False,
    'ITEMFONTCOLOR': '#000000',
    'ITEMFONTFAMILY': 'DejaVuSans',
    'ITEMFONTITALIC': False,
    'ITEMFONTSIZE': 9,
    'LAYERFONTBOLD': True,
    'LAYERFONTCOLOR': '#000000',
    'LAYERFONTFAMILY': 'DejaVuSans',
    'LAYERFONTITALIC': False,
    'LAYERFONTSIZE': 9,
    'LAYERSPACE': 4,
    'LAYERTITLE': True,
    'LAYERTITLESPACE': 4,
    'RULELABEL': True,
    'SYMBOLHEIGHT': 8,
    'SYMBOLSPACE': 2,
    'SYMBOLWIDTH': 8,
}


class Config(gws.base.legend.Config):
    """Qgis legend"""

    provider: t.Optional[provider.Config]
    """qgis provider"""
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
    """source layers to use"""


class Object(gws.base.legend.Object):
    provider: provider.Object
    sourceLayers: list[gws.SourceLayer]
    params: dict

    def configure(self):
        self.configure_provider()
        self.configure_sources()
        self.configure_params()

    def configure_provider(self):
        self.provider = provider.get_for(self)

    def configure_sources(self):
        self.configure_source_layers()

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers(self, self.provider.sourceLayers)

    def configure_params(self):
        defaults = dict(
            DPI=96,
            FORMAT=gws.lib.mime.PNG,
            # qgis legends are rendered bottom-up (rightmost first)
            # we need the straight order (leftmost first), like in the config
            LAYER=','.join(sl.name for sl in reversed(self.sourceLayers)),
            REQUEST=gws.OwsVerb.GetLegendGraphic,
            STYLE='',
            TRANSPARENT=True,
        )
        opts = gws.to_upper_dict(self.cfg('options', default={}))
        self.params = self.provider.server_params(
            gws.merge(_DEFAULT_LEGEND_PARAMS, defaults, opts))

    ##

    def render(self, args=None):
        res = self.provider.call_server(self.params, max_age=self.cacheMaxAge)
        img = gws.lib.image.from_bytes(res.content)
        return gws.LegendRenderOutput(image=img, size=img.size())
