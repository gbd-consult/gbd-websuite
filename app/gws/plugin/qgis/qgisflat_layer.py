import gws
import gws.base.layer
import gws.lib.metadata
import gws.gis.source
import gws.lib.os2
import gws.gis.ows
import gws.types as t

from . import provider


@gws.ext.config.layer('qgisflat')
class Config(gws.base.layer.Config, provider.Config):
    """Flat Qgis layer"""

    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use


@gws.ext.object.layer('qgisflat')
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

        self.sourceCrs = self.provider.crs

        if not gws.base.layer.configure.metadata(self):
            if len(self.sourceLayers) == 1:
                self.metadata = self.sourceLayers[0].metadata

        if not gws.base.layer.configure.bounds(self):
            our_crs = self.parentBounds.crs
            if self.provider.extent:
                self.bounds = gws.Bounds(
                    crs=our_crs,
                    extent=gws.gis.extent.transform(
                        self.provider.extent,
                        self.provider.crs,
                        our_crs
                    ))
            else:
                wgs_extent = gws.gis.extent.union(
                    gws.compact(sl.wgsExtent for sl in self.sourceLayers))
                self.bounds = gws.Bounds(
                    crs=our_crs,
                    extent=gws.gis.extent.transform(
                        wgs_extent,
                        gws.gis.crs.WGS84,
                        our_crs
                    ))

        if not gws.base.layer.configure.resolutions(self):
            self.resolutions = gws.gis.zoom.resolutions_from_source_layers(self.sourceLayers, self.parentResolutions)
            if not self.resolutions:
                raise gws.Error(f'layer {self.uid!r}: no matching resolutions')

        if not gws.base.layer.configure.legend(self):
            urls = gws.compact(sl.legendUrl for sl in self.sourceLayers)
            if urls:
                self.legend = self.create_child(
                    gws.ext.object.legend,
                    gws.merge(self.var('legend'), type='remote', urls=urls))
                return True

    def render(self, lri):
        return gws.base.layer.util.generic_raster_render(self, lri)

    def mapproxy_config(self, mc, options=None):
        # NB: qgis caps layers are always top-down
        layers = reversed([sl.name for sl in self.sourceLayers])

        source_uid = mc.source({
            'type': 'wms',
            'supported_srs': [self.sourceCrs.epsg],
            'forward_req_params': ['DPI__gws'],
            'concurrent_requests': self.root.app.var('server.qgis.maxRequests', default=0),
            'req': {
                'url': self.provider.url,
                'map': self.provider.path,
                'layers': ','.join(layers),
                'transparent': True,
            },
            'wms_opts': {
                'version': '1.3.0',
            },

            # add the file checksum to the config, so that the source and cache ids
            # in the mpx config are recalculated when the file changes
            '$hash': self.provider.project.sourceHash,
        })

        gws.base.layer.util.mapproxy_layer_config(self, mc, source_uid)
