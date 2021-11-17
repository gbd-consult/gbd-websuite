import gws
import gws.base.layer.image
import gws.lib.gis.source
import gws.lib.gis.util
import gws.lib.os2
import gws.lib.ows
import gws.types as t

from . import provider as provider_module


@gws.ext.Config('layer.qgisflat')
class Config(gws.base.layer.image.Config, provider_module.Config):
    """Flat Qgis layer"""

    sourceLayers: t.Optional[gws.lib.gis.source.LayerFilterConfig]  #: source layers to use


@gws.ext.Object('layer.qgisflat')
class Object(gws.base.layer.image.Object, gws.IOwsClient):
    provider: provider_module.Object
    source_crs: gws.ICrs

    def configure_source(self):
        gws.lib.ows.client.configure_layers(self, provider_module.Object, is_image=True)
        self.source_crs = gws.lib.gis.util.best_crs(
            self.provider.crs or self.crs,
            gws.lib.gis.source.supported_crs_list(self.source_layers))
        return True

    def configure_metadata(self):
        if not super().configure_metadata():
            self.set_metadata(self.var('metadata'), self.provider.metadata)
            return True

    def configure_zoom(self):
        if not super().configure_zoom():
            return gws.lib.ows.client.configure_zoom(self)

    def configure_search(self):
        if not super().configure_search():
            cfg = self.provider.search_config(self.source_layers)
            if cfg:
                self.search_providers.append(self.require_child('gws.ext.search.provider', cfg))
                return True

    def configure_legend(self):
        if not super().configure_legend():
            self.legend = gws.Legend(
                enabled=True,
                urls=[self.provider.legend_url(self.source_layers, self.var('legend.options'))]
            )
            return True

    @property
    def own_bounds(self):
        return gws.lib.gis.source.combined_bounds(self.source_layers, self.source_crs)

    def render_box(self, view, extra_params=None):
        extra_params = extra_params or {}
        if view.dpi > 90:
            extra_params['DPI__gws'] = str(view.dpi)
        return super().render_box(view, extra_params)

    def mapproxy_config(self, mc, options=None):
        # NB: qgis caps layers are always top-down
        layers = reversed([sl.name for sl in self.source_layers])

        source = mc.source({
            'type': 'wms',
            'supported_srs': [self.source_crs.epsg],
            'forward_req_params': ['DPI__gws'],
            'concurrent_requests': self.root.application.var('server.qgis.maxRequests', default=0),
            'req': {
                'url': self.provider.url,
                'map': self.provider.path,
                'layers': ','.join(layers),
                'transparent': True,
            },
            # add the file checksum to the config, so that the source and cache ids
            # in the mpx config are recalculated when the file changes
            '$hash': gws.lib.os2.file_checksum(self.provider.path)
        })

        self.mapproxy_layer_config(mc, source)
