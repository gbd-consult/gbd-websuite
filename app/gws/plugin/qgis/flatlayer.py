import gws
import gws.types as t
import gws.base.layer.image
import gws.lib.metadata
import gws.base.search
import gws.lib.mpx
import gws.lib.proj
import gws.lib.shape
import gws.lib.gis
import gws.lib.gis
import gws.lib.zoom
import gws.lib.os2
from . import provider


@gws.ext.Config('layer.qgisflat')
class Config(gws.base.layer.image.Config, provider.Config):
    """Flat Qgis layer"""

    sourceLayers: t.Optional[gws.lib.gis.SourceLayerFilter]  #: source layers to use


@gws.ext.Object('layer.qgisflat')
class Object(gws.base.layer.image.Object):
    provider: provider.Object
    source_crs: gws.Crs
    source_layers: t.List[gws.lib.gis.SourceLayer]

    def configure(self):
        if self.var('_provider'):
            self.provider = self.var('_provider')
            self.source_layers = self.var('_source_layers')
        else:
            self.provider = provider.create(self.root, self.config, shared=True)
            self.source_layers = gws.lib.gis.filter_source_layers(
                self.provider.source_layers,
                self.var('sourceLayers', default=gws.lib.gis.SourceLayerFilter(level=1)))

        if not self.source_layers:
            raise gws.Error(f'no source layers found in layer={self.uid!r}')

        if not self.has_configured_metadata:
            self.configure_metadata_from(self.provider.metadata)

        self.source_crs = gws.lib.gis.best_crs(self.map.crs, self.provider.supported_crs)

        if not self.has_configured_resolutions:
            zoom = gws.lib.zoom.config_from_source_layers(self.source_layers)
            if zoom:
                self.resolutions = gws.lib.zoom.resolutions_from_config(zoom, self.resolutions)
                self.has_configured_resolutions = True

        if not self.has_configured_search:
            cfg = self.provider.search_config(self.source_layers)
            if cfg:
                self.search_providers.append(self.require_child('gws.ext.search.provider', cfg))
                self.has_configured_search = True

        if not self.has_configured_legend:
            self.legend = gws.Legend(
                enabled=True,
                urls=[self.provider.legend_url(self.source_layers, self.var('legend.options'))]
            )
            self.has_configured_legend = True

    @property
    def own_bounds(self):
        return gws.lib.gis.bounds_from_source_layers(self.source_layers, self.source_crs)

    @property
    def description(self):
        context = {
            'layer': self,
            'service_metadata': self.provider.metadata,
            'sub_layers': self.source_layers
        }
        return self.description_template.render(context).content

    def render_box(self, rv: gws.MapRenderView, extra_params=None):
        extra_params = extra_params or {}
        if rv.dpi > 90:
            extra_params['DPI__gws'] = str(rv.dpi)
        return super().render_box(rv, extra_params)

    def render_legend_image(self, context=None):
        return self.provider.get_legend(self.source_layers, self.legend.options)

    def mapproxy_config(self, mc, options=None):
        # NB: qgis caps layers are always top-down
        layers = reversed([sl.name for sl in self.source_layers])

        source = mc.source({
            'type': 'wms',
            'supported_srs': self.provider.supported_crs,
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