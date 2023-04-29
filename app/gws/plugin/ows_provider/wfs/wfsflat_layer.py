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

gws.ext.new.layer('wfsflat')


class Config(gws.base.layer.Config):
    """Flat WFS layer."""

    provider: t.Optional[provider.Config]
    """WFS provider"""
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
    """source layers to use"""


class Object(gws.base.layer.vector.Object):
    provider: provider.Object
    sourceLayers: list[gws.SourceLayer]
    sourceCrs: gws.ICrs

    def configure(self):
        self.configure_layer()
        if len(self.sourceLayers) != 1:
            raise gws.Error(f'wfsflat requires a single source layer')

    def configure_provider(self):
        self.provider = provider.get_for(self)
        return True

    def configure_sources(self):
        if super().configure_sources():
            return True
        self.configure_source_layers()
        return True

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
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type='wfs',
            _defaultProvider=self.provider,
            _defaultSourceLayers=self.sourceLayers,
        )

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        blist = gws.compact(sl.wgsBounds for sl in self.sourceLayers)
        wgs_bounds = gws.gis.bounds.union(blist) if blist else gws.gis.crs.WGS84_BOUNDS
        self.bounds = gws.gis.bounds.transform(wgs_bounds, self.defaultBounds.crs)
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
        self.finders.append(self.configure_finder({}))
        return True

    def configure_finder(self, cfg):
        return self.create_child(
            gws.ext.object.finder,
            type='wfs',
            _defaultProvider=self.provider,
            _defaultSourceLayers=self.sourceLayers,
        )
