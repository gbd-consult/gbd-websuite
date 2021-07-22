import gws
import gws.types as t
import gws.base.layer.image
import gws.lib.metadata
import gws.base.search
import gws.config
import gws.lib.mpx
import gws.lib.proj
import gws.lib.shape
import gws.lib.gis
import gws.lib.gis
import gws.lib.zoom
import gws.lib.os2
from . import provider


class Config(gws.base.layer.image.Config):
    """WMS layer from a Qgis project"""

    path: gws.FilePath  #: qgis project path
    sourceLayers: t.Optional[gws.lib.gis.LayerFilter]  #: source layers to use


class Object(gws.base.layer.image.Object):
    @property
    def description(self):
        context = {
            'layer': self,
            'provider': self.provider.metadata
        }
        return self.description_template.render(context).content

    @property
    def own_bounds(self):
        return gws.lib.gis.bounds_from_layers(self.source_layers, self.map.crs)

    @property
    def default_search_provider(self):
        source_layers = gws.lib.gis.filter_layers(
            self.provider.source_layers,
            self.var('sourceLayers'),
            queryable_only=True
        )
        if source_layers:
            return self.root.create_object('gws.ext.search.provider.qgiswms', gws.Config(
                uid=self.uid + '.default_search',
                layer=self,
                source_layers=source_layers))

    def configure(self):
        

        self.provider: provider.Object = provider.create_shared(self.root, self.config)
        self.source_crs = gws.lib.gis.best_crs(self.map.crs, self.provider.supported_crs)

        self.source_layers = gws.lib.gis.filter_layers(
            self.provider.source_layers,
            self.var('sourceLayers'),
        )

        if not self.source_layers:
            raise gws.Error(f'no layers found in {self.uid!r}')

        self.metadata = self.configure_metadata(
            self.source_layers[0].metadata if len(self.source_layers) == 1 else None)
        self.title = self.metadata.title

        if not self.var('zoom'):
            zoom = gws.lib.zoom.config_from_source_layers(self.source_layers)
            if zoom:
                self.resolutions = gws.lib.zoom.resolutions_from_config(
                    zoom, self.resolutions)

    def render_box(self, rv: gws.MapRenderView, extra_params=None):
        extra_params = extra_params or {}
        if rv.dpi > 90:
            extra_params['DPI__gws'] = str(rv.dpi)
        return super().render_box(rv, extra_params)

    def configure_legend(self):
        return super().configure_legend() or gws.Legend(enabled=True)

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
