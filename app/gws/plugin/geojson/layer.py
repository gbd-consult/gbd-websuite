"""The ``geojson`` layer."""

import gws
import gws.base.layer
import gws.lib.shape
import gws.config.util
import gws.lib.bounds
import gws.lib.crs
import gws.lib.jsonx

from . import provider


@gws.ext.config.layer('geojson')
class Config(gws.base.layer.Config):
    """Vector layer with features from a GeoJSON file."""

    provider: provider.Config
    """GeoJSON file with the layer features."""


@gws.ext.object.layer('geojson')
class Object(gws.base.layer.vector.Object):
    """GeoJSON layer."""

    provider: provider.Object
    """The GeoJSON provider."""

    def configure(self):
        self.configure_layer()
        for rec in self.provider.load_records():
            if rec.shape:
                self.geometryType = rec.shape.type
                self.geometryCrs = rec.shape.crs
                break

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)

    def configure_extent(self):
        if super().configure_extent():
            return True
        recs = self.provider.load_records()
        if recs:
            bs = [rec.shape.bounds() for rec in recs if rec.shape]
            if bs:
                ext = gws.lib.bounds.wgs_extent(gws.lib.bounds.union(bs), pad=True)
                if ext:
                    self.wgsExtent = ext
                    return True

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        """Create a GeoJSON model that uses the provider of the layer.

        Args:
            cfg: Model configuration, or ``None`` for the default model.

        Returns:
            The model.
        """
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            _defaultProvider=self.provider
        )
    
    def configure_search(self):
        if super().configure_search():
            return True
        self.finders.append(self.create_finder(None))
        return True

    def create_finder(self, cfg):
        """Create a GeoJSON finder that uses the provider of the layer.

        Args:
            cfg: Finder configuration, or ``None`` for the default finder.

        Returns:
            The finder.
        """
        return self.create_child(
            gws.ext.object.finder,
            cfg,
            type='geojson',
            _defaultProvider=self.provider,
        )
