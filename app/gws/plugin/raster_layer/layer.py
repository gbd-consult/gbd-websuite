"""Raster image layer."""

import gws
import gws.base.layer
import gws.lib.image
import gws.config.util
import gws.gis.bounds
import gws.gis.crs
import gws.gis.zoom
import gws.gis.mapse
import gws.gis.gdalx

gws.ext.new.layer('raster')


class Config(gws.base.layer.Config):
    """Raster layer"""
    paths: list[gws.FilePath]
    """paths"""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.tile
    """layer display mode"""


# _GRID_DEFAULTS = gws.TileGrid(
#     bounds=gws.Bounds(
#         crs=gws.gis.crs.WEBMERCATOR,
#         extent=gws.gis.crs.WEBMERCATOR_SQUARE,
#     ),
#     origin=gws.Origin.nw,
#     tileSize=256,
# )

class ImageEntry(gws.Data):
    path: str
    bounds: gws.Bounds


class Object(gws.base.layer.image.Object):
    entries: list[ImageEntry]

    def configure(self):
        self.entries = []

        for path in self.cfg('paths'):
            with gws.gis.gdalx.open_raster(path) as gd:
                self.entries.append(ImageEntry(
                    path=path,
                    bounds=gd.bounds()
                ))

        if not self.entries:
            raise gws.ConfigurationError('no images found')
        for e in self.entries:
            gws.log.debug(f'entry {e.path!r} {e.bounds.extent} crs={e.bounds.crs.srid}')

        self.configure_layer()

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        b = gws.gis.bounds.union([e.bounds for e in self.entries])
        self.bounds = gws.gis.bounds.transform(b, self.parentBounds.crs)
        return True

    def configure_grid(self):
        p = self.cfg('grid', default=gws.Config())

        self.grid = gws.TileGrid(
            origin=p.origin or gws.Origin.nw,
            tileSize=p.tileSize or 256,
        )

        # if p.extent:
        #     extent = p.extent
        # else:
        #     extent = self.parentBounds.extent
        # self.grid.bounds = gws.Bounds(crs=self.bounds.crs, extent=extent)
        self.grid.bounds = self.bounds

        if p.resolutions:
            self.grid.resolutions = p.resolutions
        else:
            self.grid.resolutions = gws.gis.zoom.resolutions_from_bounds(self.grid.bounds, self.grid.tileSize)

    ##

    def props(self, user):
        p = super().props(user)
        if self.displayMode == gws.LayerDisplayMode.client:
            pass
        return p

    def render(self, lri):
        if lri.type == gws.LayerRenderInputType.box:
            pass

        if lri.type == gws.LayerRenderInputType.xyz:
            m = gws.gis.mapse.map_from_bounds(self.bounds)
            for e in self.entries:
                m.add_raster_layer(gws.gis.mapse.RasterLayerOptions(
                    path=e.path,
                    crs=e.bounds.crs,
                ))

            ext = self.bounds.extent
            w = (ext[2] - ext[0]) / (1 << lri.z)

            x0 = ext[0] + lri.x * w
            x1 = x0 + w

            y0 = ext[3] - (lri.y + 1) * w
            y1 = y0 + w

            img = m.draw(
                gws.gis.bounds.from_extent((x0, y0, x1, y1), crs=self.bounds.crs),
                (self.grid.tileSize, self.grid.tileSize)
            )

            if self.root.app.developer_option('map.annotate_render'):
                text = f'{lri.z} : {lri.x} / {lri.y}\nUID={self.uid}'
                img.add_text(text, x=5, y=5).add_box()
                content = img.to_bytes()

            return gws.LayerRenderOutput(content=img.to_bytes())
