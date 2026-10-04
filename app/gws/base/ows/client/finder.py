"""Generic OWS finder."""

import gws
import gws.base.model
import gws.base.search
import gws.config.util
import gws.gis.source


class Object(gws.base.search.finder.Object):
    """Generic OWS finder.

    Searches the queryable source layers of an OWS service provider and creates
    models bound to the provider. Protocol-specific subclasses provide the provider.
    """

    supportsGeometrySearch = True
    provider: gws.OwsServiceProvider
    """Service provider."""
    sourceLayers: list[gws.SourceLayer]
    """Source layers to search."""

    def configure(self):
        self.configure_provider()
        self.configure_sources()
        self.configure_models()
        self.configure_templates()

    def configure_provider(self):
        """Configure the service provider. Must be implemented by subclasses."""
        pass

    def configure_sources(self):
        """Configure the source layers.

        Raises:
            ``ValueError``: If the provider is ``None``.
            ``gws.Error``: If there are no queryable source layers.
        """

        gws.u.require(self.provider, 'failed to configure service provider')

        self.configure_source_layers()
        if not self.sourceLayers:
            raise gws.Error(f'no queryable layers in {self.provider}')

    def configure_source_layers(self):
        """Select the queryable source layers of the provider, or those given in the configuration.

        Returns:
            Always ``True``.
        """

        return gws.config.util.configure_source_layers_for(self, self.provider.sourceLayers, is_queryable=True)

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        """Create a model of the same type as the finder, bound to the provider and the source layers.

        Args:
            cfg: Model configuration, or ``None`` for the default model.

        Returns:
            The model object.
        """

        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            _defaultProvider=self.provider,
            _defaultSourceLayers=self.sourceLayers
        )
