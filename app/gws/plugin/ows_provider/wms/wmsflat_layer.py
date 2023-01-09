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


@gws.ext.config.layer('wmsflat')
class Config(gws.base.layer.Config, provider.Config):
    """Flat WMS layer."""

    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
    """source layers to use"""


@gws.ext.object.layer('wmsflat')
class Object(gws.base.layer.Object):
    provider: provider.Object
    sourceLayers: t.List[gws.SourceLayer]
    queryableLayers: t.List[gws.SourceLayer]
    sourceCrs: gws.ICrs

    def configure(self):
        self.configure_provider()
        self.configure_sources()

        self.configure_bounds()
        self.configure_resolutions()
        self.configure_grid()

        self.configure_legend()
        self.configure_metadata()

        self.configure_models()
        self.configure_search()
        self.configure_templates()

    def configure_provider(self):
        if self.var('_provider'):
            self.provider = self.var('_provider')
            return True
        self.provider = self.root.create_shared(provider.Object, self.config)
        return True

    def configure_sources(self):
        p = self.var('sourceLayers')
        if not p and self.var('_sourceLayers'):
            self.sourceLayers = self.var('_sourceLayers')
            self.queryableLayers = gws.gis.source.filter_layers(
                self.sourceLayers,
                gws.gis.source.LayerFilter(isQueryable=True))
        else:
            self.sourceLayers = gws.gis.source.filter_layers(
                self.provider.sourceLayers,
                gws.gis.source.LayerFilter(p, isImage=True))
            self.queryableLayers = gws.gis.source.filter_layers(
                self.provider.sourceLayers,
                gws.gis.source.LayerFilter(p, isQueryable=True))

        if not self.sourceLayers:
            raise gws.Error(f'no image layers found in {self.provider.url!r}')

        self.sourceCrs = self.provider.forceCrs or gws.gis.crs.best_match(
            self.parentBounds.crs,
            gws.gis.source.combined_crs_list(self.sourceLayers))

        return True

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        wgs_extent = gws.gis.extent.union(
            gws.compact(sl.wgsBounds.extent for sl in self.sourceLayers))
        crs = self.parentBounds.crs
        self.bounds = gws.Bounds(
            crs=crs,
            extent=gws.gis.extent.transform(wgs_extent, gws.gis.crs.WGS84, crs))
        return True

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
        urls = gws.compact(sl.legendUrl for sl in self.sourceLayers)
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

    def configure_resolutions(self):
        if super().configure_resolutions():
            return True
        self.resolutions = gws.gis.zoom.resolutions_from_source_layers(
            self.sourceLayers, self.parentResolutions)
        if self.resolutions:
            return True
        raise gws.Error(f'layer {self.uid!r}: no matching resolutions')

    def configure_models(self):
        if super().configure_models():
            return True
        if self.queryableLayers:
            self.models.append(self.create_child(
                gws.ext.object.model,
                gws.Config(type='wms', _provider=self.provider, _sourceLayers=self.queryableLayers)
            ))
        return True

    def configure_search(self):
        if super().configure_search():
            return True
        self.finders.append(self.create_child(
            gws.ext.object.finder,
            gws.Config(type='wms', _provider=self.provider, _sourceLayers=self.queryableLayers)
        ))
        return True

    ##

    def render(self, lri):
        return gws.base.layer.util.generic_raster_render(self, lri)

    def mapproxy_config(self, mc, options=None):
        layers = [sl.name for sl in self.sourceLayers if sl.name]
        if not self.var('capsLayersBottomUp'):
            layers = reversed(layers)

        op = self.provider.get_operation(gws.OwsVerb.GetMap)
        args = self.provider.prepare_operation(op)

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
