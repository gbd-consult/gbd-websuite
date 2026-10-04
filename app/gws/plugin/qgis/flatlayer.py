"""The ``qgisflat`` layer."""

from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.lib.bounds
import gws.gis.source
import gws.gis.zoom
import gws.base.metadata

from . import grabber, provider


@gws.ext.config.layer('qgisflat')
class Config(gws.base.layer.Config):
    """Layer that renders selected QGIS project layers as a single image."""

    provider: Optional[provider.Config]
    """QGIS project the layer is rendered from."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to render."""
    sqlFilters: Optional[dict]
    """SQL filters for source layers, passed to QGIS Server."""


@gws.ext.object.layer('qgisflat')
class Object(gws.base.layer.image.Object):
    """Image layer that renders selected QGIS project layers as one image.

    The image is rendered by QGIS Server. The layer also provides search,
    feature models and a legend for its source layers.
    """

    provider: provider.Object
    """QGIS provider."""
    sqlFilters: dict
    """SQL filters, mapping a source layer name (or ``*`` for all layers) to a filter expression."""
    imageLayers: list[gws.SourceLayer]
    """Source layers to render."""
    searchLayers: list[gws.SourceLayer]
    """Queryable source layers."""

    def configure(self):
        self.sqlFilters = self.cfg('sqlFilters', default={})
        self.configure_layer()

    def create_cache_name(self, cache):
        return gws.u.sha256([
            self.provider.cache_hash(),
            self.render_params(gws.LayerRenderInput()),
            vars(self.imageFormat),
            list(self.wgsExtent),
            cache.requestBuffer,
            cache.requestTiles,
        ])[: gws.base.layer.image.CACHE_NAME_LENGTH]

    def create_grabber(self, opts):
        return grabber.Object(opts, provider=self.provider, params=self.render_params(gws.LayerRenderInput()))

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)

    def configure_sources(self):
        if super().configure_sources():
            return True

        gws.u.require(self.provider, 'failed to configure service provider')

        self.configure_source_layers()
        self.imageLayers = gws.gis.source.filter_layers(self.sourceLayers, is_image=True)
        self.searchLayers = gws.gis.source.filter_layers(self.sourceLayers, is_queryable=True)

    def configure_source_layers(self):
        """Select the visible image layers of the project.

        Returns:
            Always ``True``.
        """
        return gws.config.util.configure_source_layers_for(
            self,
            self.provider.sourceLayers,
            is_image=True,
            is_visible=True,
        )

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        """Create a ``qgis`` model for the layer's queryable source layers.

        Args:
            cfg: Model configuration, or ``None`` for the default model.

        Returns:
            The model.
        """
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type='qgis',
            _defaultProvider=self.provider,
            _defaultSourceLayers=self.searchLayers,
        )

    def configure_extent(self):
        if super().configure_extent():
            return True
        self.wgsExtent = self.provider.wgsExtent
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
        self.resolutions = gws.gis.zoom.resolutions_from_source_layers(self.sourceLayers, self.cfg('_parentResolutions'), self.mapCrs)
        if not self.resolutions:
            raise gws.Error(f'layer {self.uid!r}: no matching resolutions')

    def configure_legend(self):
        # cannot use super() here, because the config must be extended with defaults
        if not self.cfg('withLegend'):
            return True
        cc = self.cfg('legend')
        options = gws.u.merge(self.provider.defaultLegendOptions, cc.options if cc else {})
        self.legend = self.create_child(
            gws.ext.object.legend,
            cc,
            type='qgis',
            options=options,
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
        return gws.config.util.configure_templates_for(self)

    def configure_search(self):
        if super().configure_search():
            return True
        if self.searchLayers:
            self.finders.append(self.create_finder(None))
            return True

    def create_finder(self, cfg):
        """Create a ``qgis`` finder for the layer's queryable source layers.

        Args:
            cfg: Finder configuration, or ``None`` for the default finder.

        Returns:
            The finder.
        """
        return self.create_child(
            gws.ext.object.finder,
            cfg,
            type='qgis',
            _defaultProvider=self.provider,
            _defaultSourceLayers=self.searchLayers,
        )

    ##

    def render_params(self, lri: gws.LayerRenderInput, parent_sql_filters: dict=None) -> dict:
        """Build the GetMap parameters for the layer.

        The parameters contain ``LAYERS`` and, if any SQL filters apply,
        ``FILTER``, on top of the extra parameters of the render input.

        Args:
            lri: Render input.
            parent_sql_filters: SQL filters of the parent layer, which override the layer's own filters.

        Returns:
            GetMap parameters.
        """
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
