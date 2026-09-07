from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.lib.bounds
import gws.gis.source
import gws.gis.zoom
import gws.base.metadata

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


class Object(gws.base.layer.image.Object):
    serviceProvider: provider.Object
    sqlFilters: dict
    imageLayers: list[gws.SourceLayer]
    searchLayers: list[gws.SourceLayer]

    def configure(self):
        self.sqlFilters = self.cfg('sqlFilters', default={})
        self.configure_layer()

    def configure_grabber(self):
        self.grabber = self.create_grabber()
        return True

    def create_cache_name(self):
        return gws.u.sha256([
            self.serviceProvider.cache_hash(),
            self.render_params(gws.LayerRenderInput()),
            self.mapCrs.srid,
            vars(self.imageFormat),
            list(self.bounds.extent),
            self.cache.requestBuffer,
            self.cache.requestTiles,
        ])[: gws.base.layer.core.CACHE_NAME_LENGTH]

    def create_grabber(self):
        return self.root.create_shared(
            grabber.Object,
            crs=self.mapCrs.srid,
            extent=self.bounds.extent,
            imageFormat=self.imageFormat,
            _defaultCache=self.cache,
            _defaultProvider=self.serviceProvider,
            _defaultParams=self.render_params(gws.LayerRenderInput()),
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

    def render_params(self, lri: gws.LayerRenderInput, parent_sql_filters: dict=None) -> dict:
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
