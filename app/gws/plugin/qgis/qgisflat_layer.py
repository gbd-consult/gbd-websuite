import gws
import gws.base.layer
import gws.gis.crs
import gws.gis.bounds
import gws.gis.extent
import gws.gis.ows
import gws.gis.source
import gws.gis.zoom
import gws.lib.metadata
import gws.lib.osx
import gws.types as t

from . import provider, render

gws.ext.new.layer('qgisflat')


class Config(gws.base.layer.Config):
    """Flat Qgis layer"""

    provider: t.Optional[provider.Config]
    """qgis provider"""
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
    """source layers to use"""


class Object(gws.base.layer.Object):
    provider: provider.Object
    sqlFilters: dict
    imageLayers: list[gws.SourceLayer]
    searchLayers: list[gws.SourceLayer]

    def configure(self):
        self.sqlFilters = self.cfg('sqlFilters', default={})

        self.configure_provider()
        self.configure_sources()
        self.configure_models()
        self.configure_bounds()
        self.configure_resolutions()
        self.configure_grid()
        self.configure_legend()
        self.configure_cache()
        self.configure_metadata()
        self.configure_templates()
        self.configure_search()

    def configure_provider(self):
        self.provider = provider.get_for(self)
        return True

    def configure_sources(self):
        if super().configure_sources():
            return True
        self.configure_source_layers()
        self.imageLayers = gws.gis.source.filter_layers(self.sourceLayers, is_image=True)
        self.searchLayers = gws.gis.source.filter_layers(self.sourceLayers, is_queryable=True)

    def configure_source_layers(self):
        p = self.cfg('sourceLayers')
        if p:
            self.sourceLayers = gws.gis.source.filter_layers(self.provider.sourceLayers, p)
            return True
        p = self.cfg('_defaultSourceLayers')
        if p:
            self.sourceLayers = p
            return True
        self.sourceLayers = gws.gis.source.filter_layers(self.provider.sourceLayers, is_visible=True)
        return True

    def configure_models(self):
        if super().configure_models():
            return True
        self.models.append(self.configure_model({}))
        return True

    def configure_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            gws.merge(
                dict(type='qgislocal', _defaultProvider=self.provider, _defaultSourceLayers=self.searchLayers),
                cfg))

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        if self.provider.bounds:
            self.bounds = gws.gis.bounds.transform(self.provider.bounds, self.defaultBounds.crs)
            return True
        wgs_bounds = gws.gis.bounds.union(gws.compact(sl.wgsBounds for sl in self.sourceLayers))
        self.bounds = gws.gis.bounds.transform(wgs_bounds, self.defaultBounds.crs)
        return True

    def configure_resolutions(self):
        if super().configure_resolutions():
            return True
        self.resolutions = gws.gis.zoom.resolutions_from_source_layers(self.sourceLayers, self.cfg('_defaultResolutions'))
        if not self.resolutions:
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
        self.legend = self.create_child(
            gws.ext.object.legend,
            self.cfg('legend'),
            type='qgis',
            _defaultProvider=self.provider,
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
        return super().configure_templates()

    def configure_search(self):
        if super().configure_search():
            return True
        if self.searchLayers:
            self.finders.append(self.configure_finder({}))
            return True

    def configure_finder(self, cfg):
        return self.create_child(
            gws.ext.object.finder,
            gws.merge(
                dict(type='qgislocal', _defaultProvider=self.provider, _defaultSourceLayers=self.searchLayers),
                cfg))

    ##

    def render(self, lri):
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

        params['LAYERS'] = layers
        if filters:
            params['FILTER'] = ';'.join(filters)

        img = render.box_to_bytes(self, lri.view, params)
        return gws.LayerRenderOutput(content=img)
