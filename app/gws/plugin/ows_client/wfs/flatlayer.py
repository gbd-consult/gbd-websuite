"""WFS flat layer."""

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


from . import provider


@gws.ext.config.layer('wfsflat')
class Config(gws.base.layer.Config):
    """Vector layer that shows the features of one WFS feature type."""

    provider: Optional[provider.Config]
    """WFS service the layer is loaded from."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Feature type to show."""


@gws.ext.object.layer('wfsflat')
class Object(gws.base.layer.vector.Object):
    """Vector layer that shows the features of a single WFS feature type.

    The features are loaded from the WFS service through a ``wfs`` model and
    can be searched with a ``wfs`` finder.
    """

    provider: provider.Object
    """WFS service provider."""
    sourceLayers: list[gws.SourceLayer]
    """Feature type shown by the layer, always a single source layer."""
    sourceCrs: gws.Crs
    """Source CRS. Not set by this class."""

    def configure(self):
        self.configure_layer()
        if len(self.sourceLayers) != 1:
            raise gws.Error(f'wfsflat requires a single source layer')

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)

    def configure_sources(self):
        if super().configure_sources():
            return True

        gws.u.require(self.provider, 'failed to configure service provider')

        self.configure_source_layers()
        return True

    def configure_source_layers(self):
        """Select the source layers from the feature types of the provider.

        Returns:
            Always ``True``.
        """
        return gws.config.util.configure_source_layers_for(self, self.provider.sourceLayers)

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        """Create a ``wfs`` model bound to the provider and the source layers of the layer.

        Args:
            cfg: Model configuration, or ``None`` for the default model.

        Returns:
            The model object.
        """
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type='wfs',
            _defaultProvider=self.provider,
            _defaultSourceLayers=self.sourceLayers
        )

    def configure_extent(self):
        if super().configure_extent():
            return True
        self.wgsExtent = gws.gis.source.combined_wgs_extent(self.sourceLayers) or self.mapCrs.wgsExtent
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
        self.finders.append(self.create_finder(None))
        return True

    def create_finder(self, cfg):
        """Create a ``wfs`` finder bound to the provider and the source layers of the layer.

        Args:
            cfg: Finder configuration, or ``None`` for the default finder.

        Returns:
            The finder object.
        """
        return self.create_child(
            gws.ext.object.finder,
            cfg,
            type='wfs',
            _defaultProvider=self.provider,
            _defaultSourceLayers=self.sourceLayers
        )
