from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.lib.bounds
import gws.lib.crs
import gws.gis.source

from . import grabber, provider

gws.ext.new.layer('wmts')


class Config(gws.base.layer.Config):
    """WMTS layer"""

    provider: provider.Config
    """WMTS provider."""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.tile
    """Layer display mode."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layer filter."""
    style: Optional[str]
    """WMTS style name."""


class Object(gws.base.layer.image.Object):
    serviceProvider: provider.Object
    sourceLayers: list[gws.SourceLayer]

    activeLayer: gws.SourceLayer
    activeStyle: gws.SourceStyle
    activeTms: gws.TileMatrixSet

    def configure(self):
        self.configure_layer()

    def configure_grabber(self):
        self.grabber = self.create_grabber()
        return True

    def create_cache_name(self):
        return gws.u.sha256([
            self.serviceProvider.cache_hash(),
            self.activeLayer.name,
            self.activeStyle.name,
            self.activeTms.identifier,
            self.mapCrs.srid,
            vars(self.imageFormat),
            list(self.bounds.extent),
            self.cache.requestBuffer,
            self.cache.requestTiles,
        ], maxlen=gws.base.layer.core.CACHE_NAME_LENGTH)

    def create_grabber(self):
        return self.root.create_shared(
            grabber.Object,
            crs=self.mapCrs.srid,
            extent=self.bounds.extent,
            imageFormat=self.imageFormat,
            _defaultCache=self.cache,
            _defaultProvider=self.serviceProvider,
            _defaultTms=self.activeTms,
            _defaultUrlTemplate=self.serviceProvider.tile_url_template(self.activeLayer, self.activeTms, self.activeStyle),
        )

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def configure_sources(self):
        if super().configure_sources():
            return True

        gws.u.require(self.serviceProvider, 'failed to configure service provider')

        self.configure_source_layers()
        if not self.sourceLayers:
            raise gws.Error('no source layers found')

        self.activeLayer = self.sourceLayers[0]
        self.configure_tms()
        self.configure_style()

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers_for(self, self.serviceProvider.sourceLayers, is_image=True)

    def configure_tms(self):
        crs = self.serviceProvider.forceCrs
        if not crs:
            crs = gws.lib.crs.best_match(self.mapCrs, [tms.crs for tms in self.activeLayer.tileMatrixSets])
        tms_list = [tms for tms in self.activeLayer.tileMatrixSets if tms.crs == crs]
        if not tms_list:
            raise gws.Error(f'no TMS for {crs} in {self.serviceProvider.url}')
        self.activeTms = tms_list[0]

    def configure_style(self):
        p = self.cfg('styleName')
        if p:
            for style in self.activeLayer.styles:
                if style.name == p:
                    self.activeStyle = style
                    return True
            raise gws.Error(f'style {p!r} not found')

        for style in self.activeLayer.styles:
            if style.isDefault:
                self.activeStyle = style
                return True

        self.activeStyle = gws.SourceStyle(name='default')
        return True

    #
    # reprojecting the world doesn't make sense, just use the map extent here
    # @TODO maybe look for more sensible grid alignment
    #
    # def configure_bounds(self):
    #     if super().configure_bounds():
    #         return True
    #     src_bounds = gws.Bounds(crs=self.activeTms.crs, extent=self.activeTms.matrices[0].extent)
    #     self.bounds = gws.lib.bounds.transform(src_bounds, self.mapCrs)
    #     return True

    def configure_resolutions(self):
        if super().configure_resolutions():
            return True
        res = [gws.lib.uom.scale_to_res(m.scale) for m in self.activeTms.matrices]
        self.resolutions = sorted(res, reverse=True)
        return True

    def configure_legend(self):
        if super().configure_legend():
            return True
        url = self.activeStyle.legendUrl
        if url:
            self.legend = self.create_child(gws.ext.object.legend, type='remote', urls=[url])
            return True

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        self.metadata = self.serviceProvider.metadata
        return True

    ##
