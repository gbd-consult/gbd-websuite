import gws
import gws.base.layer
import gws.base.legend
import gws.base.model
import gws.base.search
import gws.base.template
import gws.lib.metadata
import gws.gis.crs
import gws.gis.source
import gws.gis.zoom
import gws.gis.bounds
import gws.gis.extent
import gws.gis.ows
import gws.types as t

from . import provider

gws.ext.new.layer('wmsflat')


class Config(gws.base.layer.Config):
    """Flat WMS layer."""

    provider: t.Optional[provider.Config]
    """WMS provider"""
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
    """source layers to use"""


class Object(gws.base.layer.Object):
    provider: provider.Object
    sourceLayers: list[gws.SourceLayer]
    activeCrs: gws.ICrs

    imageLayers: list[gws.SourceLayer]
    searchLayers: list[gws.SourceLayer]

    def configure(self):
        self.configure_layer()

    def configure_provider(self):
        self.provider = provider.get_for(self)
        return True

    def configure_sources(self):
        if super().configure_sources():
            return True

        self.configure_source_layers()

        self.imageLayers = gws.gis.source.filter_layers(self.sourceLayers, is_image=True)
        self.searchLayers = gws.gis.source.filter_layers(self.sourceLayers, is_queryable=True)

        self.activeCrs = self.provider.forceCrs or gws.gis.crs.best_match(
            self.defaultBounds.crs,
            gws.gis.source.combined_crs_list(self.sourceLayers))

    def configure_source_layers(self):
        p = self.cfg('sourceLayers')
        if p:
            self.sourceLayers = gws.gis.source.filter_layers(self.provider.sourceLayers, p)
            return True
        p = self.cfg('_defaultSourceLayers')
        if p:
            self.sourceLayers = p
            return True
        self.sourceLayers = self.provider.sourceLayers
        return True

    def configure_models(self):
        if super().configure_models():
            return True
        self.models.append(self.configure_model(None))
        return True

    def configure_model(self, cfg):
        return self.create_child(gws.ext.object.model, cfg, type='wms', _defaultProvider=self.provider, _defaultSourceLayers=self.searchLayers)

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        blist = gws.compact(sl.wgsBounds for sl in self.imageLayers)
        wgs_bounds = gws.gis.bounds.union(blist) if blist else gws.gis.crs.WGS84_BOUNDS
        self.bounds = gws.gis.bounds.transform(wgs_bounds, self.defaultBounds.crs)
        return True

    def configure_resolutions(self):
        if super().configure_resolutions():
            return True
        self.resolutions = gws.gis.zoom.resolutions_from_source_layers(
            self.sourceLayers, self.cfg('_defaultResolutions'))
        if self.resolutions:
            return True
        raise gws.Error(f'layer {self.uid!r}: no matching resolutions')

    def configure_grid(self):
        if super().configure_grid():
            return True
        self.grid = gws.TileGrid(
            corner=gws.Corner.nw,
            tileSize=256,
            bounds=self.bounds,
            resolutions=self.resolutions)
        return True

    def configure_legend(self):
        if super().configure_legend():
            return True
        urls = gws.compact(sl.legendUrl for sl in self.imageLayers)
        if urls:
            self.legend = self.create_child(
                gws.ext.object.legend,
                gws.Config(type='remote', urls=urls))
            return True

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        if len(self.sourceLayers) == 1:
            self.metadata = self.sourceLayers[0].metadata
            return True

    def configure_templates(self):
        return super().configure_templates()

    def configure_search(self):
        if super().configure_search():
            return True
        if self.searchLayers:
            self.finders.append(self.configure_finder({}))
            return True

    def configure_finder(self, cfg):
        return self.create_child(gws.ext.object.finder, type='wms', _defaultProvider=self.provider, _defaultSourceLayers=self.searchLayers)

    ##

    def render(self, lri):
        return gws.base.layer.util.mpx_raster_render(self, lri)

    def mapproxy_config(self, mc, options=None):
        layers = [sl.name for sl in self.imageLayers]
        if not self.provider.capsLayersBottomUp:
            layers = reversed(layers)

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
            'supported_srs': [self.activeCrs.epsg],
            'concurrent_requests': self.provider.maxRequests,
            'req': req,
            'wms_opts': {
                'version': self.provider.version,
            }
        }))

        gws.base.layer.util.mapproxy_layer_config(self, mc, source_uid)
