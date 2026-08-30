from typing import Optional

import gws
import gws.base.layer
import gws.base.legend
import gws.base.model
import gws.base.search
import gws.base.template
import gws.config.util
import gws.base.metadata
import gws.lib.crs
import gws.gis.source
import gws.gis.zoom
import gws.lib.bounds
import gws.lib.extent
import gws.lib.grid

from . import grabber, provider

gws.ext.new.layer('wmsflat')


class Config(gws.base.layer.Config):
    """Flat WMS layer."""

    provider: Optional[provider.Config]
    """WMS provider."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to use."""
    devGrabber: bool = False
    """Use the grabber instead of MapProxy."""


class Object(gws.base.layer.image.Object):
    serviceProvider: provider.Object
    sourceLayers: list[gws.SourceLayer]
    sourceCrs: gws.Crs

    imageLayers: list[gws.SourceLayer]
    searchLayers: list[gws.SourceLayer]
    devGrabber: Optional[grabber.Object]

    def configure(self):
        self.configure_layer()
        self.devGrabber = None
        if self.cfg('devGrabber'):
            self.devGrabber = self.create_grabber()

    def create_grabber(self):
        cache = self.cache or gws.LayerCache(maxAge=0, maxLevel=0)
        uid = 'grabber_' + gws.u.sha256([
            self.serviceProvider.uid,
            [sl.name for sl in self.imageLayers],
            self.sourceCrs.srid,
            self.mapCrs.srid,
            vars(self.imageFormat),
            list(self.bounds.extent),
            cache.maxAge or 0,
            cache.maxLevel or 0,
            cache.requestTiles or 0,
            cache.requestBuffer or 0,
        ])
        return self.root.create_shared(
            grabber.Object,
            uid=uid,
            crs=self.mapCrs.srid,
            extent=self.bounds.extent,
            imageFormat=self.imageFormat,
            blockSize=cache.requestTiles or grabber.DEFAULT_BLOCK_SIZE,
            cacheMaxAge=cache.maxAge or 0,
            cacheMaxLevel=cache.maxLevel or 0,
            edgeBuffer=cache.requestBuffer or 0,
            _defaultProvider=self.serviceProvider,
            _defaultSourceLayers=self.imageLayers,
            _defaultSourceCrs=self.sourceCrs,
        )

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def configure_sources(self):
        if super().configure_sources():
            return True

        gws.u.require(self.serviceProvider, 'failed to configure service provider')

        self.configure_source_layers()
        if not self.sourceLayers:
            raise gws.Error(f'layer {self!r}: no source layers found for {self.serviceProvider.url!r}')

        self.imageLayers = gws.gis.source.filter_layers(self.sourceLayers, is_image=True)
        self.searchLayers = gws.gis.source.filter_layers(self.sourceLayers, is_queryable=True)

        self.sourceCrs = self.serviceProvider.forceCrs or gws.lib.crs.best_match(
            self.mapCrs,
            gws.gis.source.combined_crs_list(self.sourceLayers))

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers_for(self, self.serviceProvider.sourceLayers)

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type='wms',
            _defaultProvider=self.serviceProvider,
            _defaultSourceLayers=self.sourceLayers
        )

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        self.bounds = gws.gis.source.combined_bounds(self.imageLayers, self.mapCrs) or self.mapCrs.bounds
        return True

    def configure_resolutions(self):
        if super().configure_resolutions():
            return True
        self.resolutions = gws.gis.zoom.resolutions_from_source_layers(
            self.sourceLayers, self.cfg('_parentResolutions'))
        if self.resolutions:
            return True
        raise gws.Error(f'layer {self!r}: no matching resolutions')

    def configure_grid(self):
        if super().configure_grid():
            return True
        self.grid = gws.TileGrid(
            origin=gws.Origin.nw,
            tileSize=256,
            bounds=self.bounds,
            resolutions=self.resolutions)
        return True

    def configure_legend(self):
        if super().configure_legend():
            return True
        urls = gws.u.compact(sl.legendUrl for sl in self.imageLayers)
        if urls:
            self.legend = self.create_child(gws.ext.object.legend, type='remote', urls=urls)
            return True

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        if len(self.sourceLayers) == 1:
            self.metadata = self.sourceLayers[0].metadata
            return True

    def configure_search(self):
        if super().configure_search():
            return True
        if self.searchLayers:
            self.finders.append(self.create_finder(None))
            return True

    def create_finder(self, cfg):
        return self.create_child(
            gws.ext.object.finder,
            cfg,
            type='wms',
            _defaultProvider=self.serviceProvider,
            _defaultSourceLayers=self.searchLayers
        )

    ##

    def props(self, user):
        p = super().props(user)
        if self.devGrabber:
            g = self.devGrabber.grid
            zmax = gws.lib.grid.level_for_resolution(g, min(self.resolutions))
            p.grid = gws.base.layer.core.GridProps(
                origin=gws.Origin.nw,
                extent=g.extent,
                resolutions=[gws.lib.grid.resolution_for_level(g, z) for z in range(zmax + 1)],
                tileSize=g.tileSize,
            )
        return p

    def render(self, lri):
        if self.devGrabber:
            return self.render_with_grabber(lri)
        return gws.base.layer.util.mpx_raster_render(self, lri)

    def render_with_grabber(self, lri):
        if lri.type == gws.LayerRenderInputType.xyz:
            return gws.LayerRenderOutput(content=self.devGrabber.get_tile((lri.x, lri.y, lri.z)))
        if lri.type == gws.LayerRenderInputType.box:
            def get_box(bounds, width, height):
                return self.devGrabber.get_box(bounds.extent, width, height)

            content = gws.base.layer.util.generic_render_box(self, lri, get_box)
            return gws.LayerRenderOutput(content=content)

    def mapproxy_config(self, mc, options=None):
        # NB reversed: see the note in plugin/ows_client/wms/provider.py
        layers = reversed([sl.name for sl in self.imageLayers])
        op = self.serviceProvider.get_operation(gws.OwsVerb.GetMap)
        args = self.serviceProvider.prepare_operation(op)

        req = gws.u.merge(
            args.params,
            transparent=True,
            layers=','.join(layers),
            url=args.url,
        )

        src = gws.u.compact({
            'type': 'wms',
            'supported_srs': [self.sourceCrs.epsg],
            'concurrent_requests': self.serviceProvider.maxRequests,
            'req': req,
            'wms_opts': {
                'version': self.serviceProvider.version,
            }
        })

        if args.headers:
            src['http'] = {'headers': args.headers}

        source_uid = mc.source(src)

        gws.base.layer.util.mapproxy_layer_config(self, mc, source_uid)
