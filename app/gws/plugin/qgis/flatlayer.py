from typing import Optional

import gws
import gws.base.grabber.box
import gws.base.layer
import gws.config.util
import gws.lib.crs
import gws.lib.bounds
import gws.lib.extent
import gws.gis.source
import gws.gis.zoom
import gws.base.metadata
import gws.lib.grid
import gws.lib.osx

from . import grabber, provider

gws.ext.new.layer('qgisflat')


class Config(gws.base.layer.Config):
    """Flat Qgis layer"""

    provider: Optional[provider.Config]
    """Qgis provider."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to use."""
    sqlFilters: Optional[dict]
    """Per-layer sql filters."""
    devGrabber: bool = False
    """Use the grabber instead of the direct render path."""


class Object(gws.base.layer.image.Object):
    serviceProvider: provider.Object
    sqlFilters: dict
    imageLayers: list[gws.SourceLayer]
    searchLayers: list[gws.SourceLayer]
    devGrabber: Optional[grabber.Object]

    def configure(self):
        self.sqlFilters = self.cfg('sqlFilters', default={})
        self.configure_layer()
        self.devGrabber = None
        if self.cfg('devGrabber'):
            self.devGrabber = self.create_grabber()

    def create_grabber(self):
        cache = self.cache or gws.LayerCache(maxAge=0, maxLevel=0)
        params = self.get_render_params(gws.LayerRenderInput())
        uid = 'grabber_' + gws.u.sha256([
            self.serviceProvider.uid,
            params,
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
            blockSize=cache.requestTiles or gws.base.grabber.box.DEFAULT_BLOCK_SIZE,
            cacheMaxAge=cache.maxAge or 0,
            cacheMaxLevel=cache.maxLevel or 0,
            edgeBuffer=cache.requestBuffer or 0,
            _defaultProvider=self.serviceProvider,
            _defaultParams=params,
        )

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def configure_sources(self):
        if super().configure_sources():
            return True

        gws.u.require(self.serviceProvider, 'failed to configure service provider')

        self.configure_source_layers()
        self.imageLayers = gws.gis.source.filter_layers(self.sourceLayers, is_image=True)
        self.searchLayers = gws.gis.source.filter_layers(self.sourceLayers, is_queryable=True)

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers_for(
            self,
            self.serviceProvider.sourceLayers,
            is_image=True,
            is_visible=True,
        )

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type='qgis',
            _defaultProvider=self.serviceProvider,
            _defaultSourceLayers=self.searchLayers,
        )

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        self.bounds = gws.lib.bounds.transform(self.serviceProvider.bounds, self.mapCrs)
        return True

    def configure_zoom_bounds(self):
        if super().configure_zoom_bounds():
            return True
        b = gws.gis.source.combined_bounds(self.sourceLayers, self.mapCrs)
        if b:
            self.zoomBounds = b
            return True

    def configure_resolutions(self):
        if super().configure_resolutions():
            return True
        self.resolutions = gws.gis.zoom.resolutions_from_source_layers(self.sourceLayers, self.cfg('_parentResolutions'))
        if not self.resolutions:
            raise gws.Error(f'layer {self.uid!r}: no matching resolutions')

    def configure_grid(self):
        if super().configure_grid():
            return True
        self.grid = gws.TileGrid(
            origin=gws.Origin.nw,
            tileSize=256,
            bounds=self.bounds,
            resolutions=self.resolutions,
        )
        return True

    def configure_legend(self):
        # cannot use super() here, because the config must be extended with defaults
        if not self.cfg('withLegend'):
            return True
        cc = self.cfg('legend')
        options = gws.u.merge(self.serviceProvider.defaultLegendOptions, cc.options if cc else {})
        self.legend = self.create_child(
            gws.ext.object.legend,
            cc,
            type='qgis',
            options=options,
            _defaultProvider=self.serviceProvider,
            _defaultSourceLayers=self.imageLayers,
        )
        return True

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        if len(self.sourceLayers) == 1:
            self.metadata = self.sourceLayers[0].metadata
            return True

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self)

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
            type='qgis',
            _defaultProvider=self.serviceProvider,
            _defaultSourceLayers=self.searchLayers,
        )

    ##

    def get_render_params(self, lri: gws.LayerRenderInput, parent_sql_filters: dict=None) -> dict:
        params = dict(lri.extraParams or {})
        
        layers = [sl.name for sl in self.imageLayers]
        # NB reversed: see the note in plugin/ows_client/wms/provider.py
        params['LAYERS'] = list(reversed(layers))
        
        filters = []
        sqlf = gws.u.merge({}, self.sqlFilters, parent_sql_filters)
        for name in layers:
            flt = sqlf.get(name) or sqlf.get('*')
            if flt:
                filters.append(name + ': ' + flt)
        if filters:
            params['FILTER'] = ';'.join(filters)

        return params

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

        if lri.type != gws.LayerRenderInputType.box:
            return

        params = self.get_render_params(lri)

        def get_box(bounds, width, height):
            return self.serviceProvider.get_map(self, bounds, width, height, params)

        content = gws.base.layer.util.generic_render_box(self, lri, get_box)
        return gws.LayerRenderOutput(content=content)

    def render_with_grabber(self, lri):
        if lri.type == gws.LayerRenderInputType.xyz:
            return gws.LayerRenderOutput(content=self.devGrabber.get_tile((lri.x, lri.y, lri.z)))
        if lri.type == gws.LayerRenderInputType.box:
            def get_box(bounds, width, height):
                return self.devGrabber.get_box(bounds.extent, width, height)

            content = gws.base.layer.util.generic_render_box(self, lri, get_box)
            return gws.LayerRenderOutput(content=content)
