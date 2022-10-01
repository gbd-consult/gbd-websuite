import gws
import gws.base.layer
import gws.base.legend
import gws.lib.metadata
import gws.gis.source
import gws.gis.bounds
import gws.gis.extent
import gws.gis.ows
import gws.types as t

from . import provider


@gws.ext.config.layer('wmsflat')
class Config(gws.base.layer.Config, provider.Config):
    """Flat WMS layer."""

    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use


@gws.ext.object.layer('wmsflat')
class Object(gws.base.layer.Object, gws.IOwsClient):
    provider: provider.Object
    sourceCrs: gws.ICrs

    def configure(self):
        self.provider = self.var('_provider') or self.root.create_shared(provider.Object, self.config)

        self.sourceLayers = self.var('_sourceLayers')
        if not self.sourceLayers:
            slf = gws.merge(
                gws.gis.source.LayerFilter(isImage=True),
                self.var('sourceLayers'))
            self.sourceLayers = gws.gis.source.filter_layers(self.provider.sourceLayers, slf)
        if not self.sourceLayers:
            raise gws.Error(f'no source layers found in {self.provider.url!r}')

        self.sourceCrs = gws.gis.crs.best_match(
            self.provider.forceCrs or self.parentBounds.crs,
            gws.gis.source.combined_crs_list(self.sourceLayers))

        if not gws.base.layer.configure.metadata(self):
            if len(self.sourceLayers) == 1:
                self.metadata = self.sourceLayers[0].metadata

        if not gws.base.layer.configure.bounds(self):
            wgs_extent = gws.gis.extent.union(
                gws.compact(sl.wgsExtent for sl in self.sourceLayers))
            crs = self.parentBounds.crs
            self.bounds = gws.Bounds(
                crs=crs,
                extent=gws.gis.extent.transform(wgs_extent, gws.gis.crs.WGS84, crs))

        if not gws.base.layer.configure.resolutions(self):
            self.resolutions = gws.gis.zoom.resolutions_from_source_layers(self.sourceLayers, self.parentResolutions)
            if not self.resolutions:
                raise gws.Error(f'layer {self.uid!r}: no matching resolutions')

        if not gws.base.layer.configure.legend(self):
            urls = gws.compact(sl.legendUrl for sl in self.sourceLayers)
            if urls:
                self.legend = self.create_child(
                    gws.ext.object.legend,
                    gws.Config(type='remote', urls=urls))
                return True

    # def configure_search(self):
    #     if not super().configure_search():
    #         queryable_layers = gws.gis.source.filter_layers(
    #             self.sourceLayers,
    #             gws.gis.source.LayerFilter(isQueryable=True)
    #         )
    #         if queryable_layers:
    #             self.searchMgr.add_finder(gws.Config(
    #                 type='wms',
    #                 _provider=self.provider,
    #                 _sourceLayers=queryable_layers
    #             ))
    #             return True

    def render(self, lri):
        return gws.base.layer.util.generic_raster_render(self, lri)

    def mapproxy_config(self, mc, options=None):
        layers = [sl.name for sl in self.sourceLayers if sl.name]
        if not self.var('capsLayersBottomUp'):
            layers = reversed(layers)

        op = self.provider.operation(gws.OwsVerb.GetMap)
        args = self.provider.request_args_for_operation(op)

        req = gws.merge({}, args.params, {
            'transparent': True,
            'layers': ','.join(layers),
            'url': args.url,
        })

        source_uid = mc.source(gws.compact({
            'type': 'wms',
            'supported_srs': [self.sourceCrs.epsg],
            'concurrent_requests': self.var('maxRequests'),
            'req': req,
            'wms_opts': {
                'version': self.provider.version,
            }
        }))

        gws.base.layer.util.mapproxy_layer_config(self, mc, source_uid)
