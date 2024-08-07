from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.gis.bounds
import gws.gis.crs
import gws.gis.source
import gws.gis.zoom
import gws.lib.uom as units

from . import provider

gws.ext.new.layer('wmts')


class Config(gws.base.layer.Config):
    """WMTS layer"""
    provider: provider.Config
    """WMTS provider"""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.tile
    """layer display mode"""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """source layer filter"""
    style: Optional[str]
    """WMTS style name"""


class Object(gws.base.layer.image.Object):
    serviceProvider: provider.Object
    sourceLayers: list[gws.SourceLayer]

    activeLayer: gws.SourceLayer
    activeStyle: gws.SourceStyle
    activeTms: gws.TileMatrixSet

    def configure(self):
        self.configure_layer()

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def configure_sources(self):
        if super().configure_sources():
            return True

        self.configure_source_layers()
        self.activeLayer = self.sourceLayers[0]
        self.configure_tms()
        self.configure_style()

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers_for(self, self.serviceProvider.sourceLayers, is_image=True)

    def configure_tms(self):
        crs = self.serviceProvider.forceCrs
        if not crs:
            crs = gws.gis.crs.best_match(self.mapCrs, [tms.crs for tms in self.activeLayer.tileMatrixSets])
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
    #     self.bounds = gws.gis.bounds.transform(src_bounds, self.mapCrs)
    #     return True

    def configure_resolutions(self):
        if super().configure_resolutions():
            return True
        res = [gws.lib.uom.scale_to_res(m.scale) for m in self.activeTms.matrices]
        self.resolutions = sorted(res, reverse=True)
        return True

    def configure_grid(self):
        p = self.cfg('grid', default=gws.Config())
        self.grid = gws.TileGrid(
            origin=p.origin or gws.Origin.nw,
            tileSize=p.tileSize or self.activeTms.matrices[0].tileWidth,
        )
        if p.extent:
            self.grid.bounds = gws.Bounds(crs=self.mapCrs, extent=p.extent)
        elif self.activeTms.crs == self.mapCrs:
            self.grid.bounds = gws.Bounds(crs=self.mapCrs, extent=self.activeTms.matrices[0].extent)
        else:
            self.grid.bounds = self.bounds

        if p.resolutions:
            self.grid.resolutions = p.resolutions
        else:
            self.grid.resolutions = gws.gis.zoom.resolutions_from_bounds(self.grid.bounds, self.grid.tileSize)

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

    def mapproxy_config(self, mc):
        url = self.serviceProvider.tile_url_template(self.activeLayer, self.activeTms, self.activeStyle)

        # mapproxy encoding

        url = url.replace('{TileMatrix}', '%(z)02d')
        url = url.replace('{TileCol}', '%(x)d')
        url = url.replace('{TileRow}', '%(y)d')

        source_grid = self.serviceProvider.grid_for_tms(self.activeTms)

        if source_grid.origin == gws.Origin.nw:
            origin = 'nw'
        elif source_grid.origin == gws.Origin.sw:
            origin = 'sw'
        else:
            raise gws.Error(f'invalid grid origin {source_grid.origin!r}')

        source_grid_uid = mc.grid(gws.u.compact({
            'origin': origin,
            'srs': source_grid.bounds.crs.epsg,
            'bbox': source_grid.bounds.extent,
            'res': source_grid.resolutions,
            'tile_size': [source_grid.tileSize, source_grid.tileSize],
        }))

        src_uid = gws.base.layer.util.mapproxy_back_cache_config(self, mc, url, source_grid_uid)
        gws.base.layer.util.mapproxy_layer_config(self, mc, src_uid)

    ##

    def render(self, lri):
        return gws.base.layer.util.mpx_raster_render(self, lri)
