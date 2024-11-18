from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.gis.crs
import gws.gis.bounds
import gws.gis.extent
import gws.gis.source
import gws.gis.zoom
import gws.lib.metadata
import gws.lib.osx

from . import provider

gws.ext.new.layer('qgisflat')


class Config(gws.base.layer.Config):
    """Flat Qgis layer"""

    provider: Optional[provider.Config]
    """qgis provider"""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """source layers to use"""


class Object(gws.base.layer.image.Object):
    serviceProvider: provider.Object
    sqlFilters: dict
    imageLayers: list[gws.SourceLayer]
    searchLayers: list[gws.SourceLayer]

    def configure(self):
        self.sqlFilters = self.cfg('sqlFilters', default={})
        self.configure_layer()

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def configure_sources(self):
        if super().configure_sources():
            return True
        self.configure_source_layers()
        self.imageLayers = gws.gis.source.filter_layers(self.sourceLayers, is_image=True)
        self.searchLayers = gws.gis.source.filter_layers(self.sourceLayers, is_queryable=True)

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers_for(
            self,
            self.serviceProvider.sourceLayers,
            is_image=True,
            is_visible=True
        )

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type='qgis',
            _defaultProvider=self.serviceProvider,
            _defaultSourceLayers=self.searchLayers
        )

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        self.bounds = gws.gis.bounds.transform(self.serviceProvider.bounds, self.mapCrs)
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
            resolutions=self.resolutions)
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
            _defaultSourceLayers=self.searchLayers
        )

    ##

    def render(self, lri):
        if lri.type != gws.LayerRenderInputType.box:
            return

        params = dict(lri.extraParams or {})
        all_names = [sl.name for sl in self.imageLayers]
        req_names = params.pop('layers', all_names)
        req_filters = params.pop('filters', self.sqlFilters)

        layers = []
        filters = []

        for name in req_names:
            if name not in all_names:
                gws.log.warning(f'invalid layer name {name!r}')
                continue
            layers.append(name)
            flt = req_filters.get(name) or req_filters.get('*')
            if flt:
                filters.append(name + ': ' + flt)

        # NB reversed: see the note in plugin/ows_client/wms/provider.py
        params['LAYERS'] = list(reversed(layers))
        if filters:
            params['FILTER'] = ';'.join(filters)

        def get_box(bounds, width, height):
            return self.serviceProvider.get_map(self, bounds, width, height, params)

        content = gws.base.layer.util.generic_render_box(self, lri, get_box)
        return gws.LayerRenderOutput(content=content)
