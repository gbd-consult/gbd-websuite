import gws
import gws.base.layer
import gws.base.legend
import gws.base.model
import gws.base.search
import gws.base.template
import gws.config.util
import gws.lib.metadata
import gws.gis.crs
import gws.gis.source
import gws.gis.zoom
import gws.gis.bounds
import gws.gis.extent
import gws.types as t

from . import provider

gws.ext.new.layer('wmsflat')


class Config(gws.base.layer.Config):
    """Flat WMS layer."""

    provider: t.Optional[provider.Config]
    """WMS provider"""
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
    """source layers to use"""


class Object(gws.base.layer.image.Object):
    provider: provider.Object
    sourceLayers: list[gws.SourceLayer]
    sourceCrs: gws.ICrs

    imageLayers: list[gws.SourceLayer]
    searchLayers: list[gws.SourceLayer]

    def configure(self):
        self.configure_layer()

    def configure_provider(self):
        self.provider = provider.get_for(self)
        if not self.provider:
            raise gws.Error(f'layer {self!r}: no provider found')
        return True

    def configure_sources(self):
        if super().configure_sources():
            return True

        self.configure_source_layers()
        if not self.sourceLayers:
            raise gws.Error(f'layer {self!r}: no source layers found for {self.provider.url!r}')

        self.imageLayers = gws.gis.source.filter_layers(self.sourceLayers, is_image=True)
        self.searchLayers = gws.gis.source.filter_layers(self.sourceLayers, is_queryable=True)

        self.sourceCrs = self.provider.forceCrs or gws.gis.crs.best_match(
            self.mapCrs,
            gws.gis.source.combined_crs_list(self.sourceLayers))

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers(self, self.provider.sourceLayers)

    def configure_models(self):
        return gws.config.util.configure_models(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type='wms',
            _defaultProvider=self.provider,
            _defaultSourceLayers=self.sourceLayers
        )

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        blist = gws.compact(sl.wgsBounds for sl in self.imageLayers)
        wgs_bounds = gws.gis.bounds.union(blist) if blist else gws.gis.crs.WGS84_BOUNDS
        self.bounds = gws.gis.bounds.transform(wgs_bounds, self.mapCrs)
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
        urls = gws.compact(sl.legendUrl for sl in self.imageLayers)
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
            _defaultProvider=self.provider,
            _defaultSourceLayers=self.searchLayers
        )

    ##

    def render(self, lri):
        return gws.base.layer.util.mpx_raster_render(self, lri)

    def mapproxy_config(self, mc, options=None):
        # NB reversed: see the note in plugin/ows_client/wms/provider.py
        layers = reversed([sl.name for sl in self.imageLayers])
        op = self.provider.get_operation(gws.OwsVerb.GetMap)
        args = self.provider.prepare_operation(op)

        req = gws.merge(
            args.params,
            transparent=True,
            layers=','.join(layers),
            url=args.url,
        )

        source_uid = mc.source(gws.compact({
            'type': 'wms',
            'supported_srs': [self.sourceCrs.epsg],
            'concurrent_requests': self.provider.maxRequests,
            'req': req,
            'wms_opts': {
                'version': self.provider.version,
            }
        }))

        gws.base.layer.util.mapproxy_layer_config(self, mc, source_uid)
