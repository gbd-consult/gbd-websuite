from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.lib.bounds
import gws.lib.crs
import gws.lib.extent
import gws.lib.uom
import gws.gis.source
import gws.gis.zoom

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

    def create_cache_name(self, cache):
        return gws.u.sha256(
            [
                self.serviceProvider.cache_hash(),
                self.activeLayer.name,
                self.activeStyle.name,
                self.activeTms.identifier,
                vars(self.imageFormat),
                list(self.wgsExtent),
                cache.requestBuffer,
                cache.requestTiles,
            ]
        )[: gws.base.layer.core.CACHE_NAME_LENGTH]

    def create_grabber(self, opts):
        return grabber.Object(
            opts,
            serviceProvider=self.serviceProvider,
            tms=self.activeTms,
            urlTemplate=self.serviceProvider.tile_url_template(self.activeLayer, self.activeTms, self.activeStyle),
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

    def configure_extent(self):
        if super().configure_extent():
            return True
        tms_crs = self.activeTms.crs
        ext = gws.lib.extent.transform_to_wgs(self.activeTms.matrices[0].extent, tms_crs)
        if gws.lib.extent.is_valid_wgs(ext):
            self.wgsExtent = tms_crs.clip_wgs_extent(ext) or tms_crs.wgsMaxExtent
        else:
            self.wgsExtent = tms_crs.wgsMaxExtent

        layer_ext = gws.gis.source.combined_wgs_extent([self.activeLayer])
        if layer_ext:
            ext = gws.lib.extent.intersection(self.wgsExtent, layer_ext)
            if ext and gws.lib.extent.is_valid_wgs(ext):
                self.wgsExtent = ext

        return True

    def configure_resolutions(self):
        if super().configure_resolutions():
            return True
        scales = [m.scale for m in self.activeTms.matrices]
        self.resolutions = gws.gis.zoom.resolutions_from_scale_range(
            min(scales),
            max(scales),
            self.cfg('_parentResolutions'),
            self.mapCrs,
        )
        if self.resolutions:
            return True
        raise gws.Error(f'layer {self!r}: no matching resolutions')

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
