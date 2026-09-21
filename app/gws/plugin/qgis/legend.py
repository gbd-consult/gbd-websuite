"""QGIS legend."""

from typing import Optional

import gws
import gws.base.legend
import gws.config.util
import gws.lib.mime
import gws.gis.source
import gws.lib.image


from . import provider


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


@gws.ext.config.legend('qgis')
class Config(gws.base.legend.Config):
    """Qgis legend"""

    provider: Optional[provider.Config]
    """Qgis provider."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to use."""


@gws.ext.object.legend('qgis')
class Object(gws.base.legend.Object):
    provider: provider.Object
    sourceLayers: list[gws.SourceLayer]
    params: dict

    def configure(self):
        self.configure_provider()
        self.configure_sources()
        self.configure_params()

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)

    def configure_sources(self):
        gws.u.require(self.provider, 'failed to configure service provider')
        self.configure_source_layers()

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers_for(self, self.provider.sourceLayers)

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
        opts = gws.u.to_upper_dict(self.cfg('options', default={}))
        self.params = self.provider.server_params(gws.u.merge(_DEFAULT_LEGEND_PARAMS, defaults, opts))

    ##

    def render(self, args=None):
        def _get():
            return self.provider.call_server(self.params).content

        content = gws.u.get_cached_object(
            f'legend_{gws.u.sha256([self.provider.url, self.params])}',
            self.cacheMaxAge,
            _get,
        )
        img = gws.lib.image.from_bytes(content)
        return gws.LegendRenderOutput(image=img, size=img.size(), mime=gws.lib.mime.PNG)
