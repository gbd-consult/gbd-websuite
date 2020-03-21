import gws

import gws.common.layer
import gws.common.metadata
import gws.common.search
import gws.config
import gws.gis.mpx
import gws.gis.proj
import gws.gis.shape
import gws.gis.source
import gws.gis.util
import gws.gis.zoom
import gws.tools.os2

import gws.types as t

from . import provider, wmssearch


class Config(gws.common.layer.ImageConfig):
    """WMS layer from a Qgis project"""

    path: t.FilePath  #: qgis project path
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]  #: source layers to use


class Object(gws.common.layer.Image):
    def configure(self):
        super().configure()

        self.provider: provider.Object = provider.create_shared(self, self.config)
        self.source_crs = gws.gis.util.best_crs(self.map.crs, self.provider.supported_crs)

        self.source_layers = gws.gis.source.filter_layers(
            self.provider.source_layers,
            self.var('sourceLayers'),
        )

        if not self.source_layers:
            raise gws.Error(f'no layers found in {self.uid!r}')

        self.meta = self.configure_metadata(
            self.source_layers[0].meta if len(self.source_layers) == 1 else None)
        self.title = self.meta.title

        if not self.var('zoom'):
            zoom = gws.gis.zoom.config_from_source_layers(self.source_layers)
            if zoom:
                self.resolutions = gws.gis.zoom.resolutions_from_config(
                    zoom, self.resolutions)

        self.has_legend = self.var('legend.enabled')

    @property
    def description(self):
        ctx = {
            'layer': self,
            'provider': self.provider.meta
        }
        return self.description_template.render(ctx).content
        #
        # sub_layers = self.source_layers
        # if len(sub_layers) == 1 and gws.get(sub_layers[0], 'title') == self.title:
        #     sub_layers = []
        #
        # return super().description(gws.defaults(
        #     options,
        #     sub_layers=sub_layers))

    @property
    def own_bounds(self):
        return gws.gis.source.bounds_from_layers(self.source_layers, self.map.crs)

    @property
    def default_search_provider(self):
        source_layers = gws.gis.source.filter_layers(
            self.provider.source_layers,
            self.var('sourceLayers'),
            queryable_only=True
        )
        if source_layers:
            return self.root.create_object('gws.ext.search.provider.qgiswms', t.Config(
                uid=self.uid + '.default_search',
                layer=self,
                source_layers=source_layers))

    def render_box(self, rv: t.RenderView, client_params=None):
        forward = {}

        cache_uid = self.uid

        if not self.has_cache:
            cache_uid = self.uid + '_NOCACHE'

        if rv.dpi > 90:
            forward['DPI__gws'] = str(rv.dpi)
            cache_uid = self.uid + '_NOCACHE'

        return gws.gis.mpx.wms_request(
            cache_uid,
            rv.bounds,
            rv.size_px[0],
            rv.size_px[1],
            forward)

    def render_legend(self):
        if not self.has_legend:
            return
        if self.legend_url:
            return super().render_legend()
        return self.provider.get_legend(self.source_layers)

    def mapproxy_config(self, mc, options=None):
        # NB: qgis caps layers are always top-down
        layers = reversed([sl.name for sl in self.source_layers])

        source = mc.source({
            'type': 'wms',
            'supported_srs': self.provider.supported_crs,
            'forward_req_params': ['DPI__gws'],
            'concurrent_requests': self.root.var('server.qgis.maxRequests', default=0),
            'req': {
                'url': self.provider.url,
                'map': self.provider.path,
                'layers': ','.join(layers),
                'transparent': True,
            },
            # add the file checksum to the config, so that the source and cache ids
            # in the mpx config are recalculated when the file changes
            '$hash': gws.tools.os2.file_checksum(self.provider.path)
        })

        self.mapproxy_layer_config(mc, source)
