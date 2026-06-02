"""Exporter for the ESRI Shapefile format."""


import gws
import gws.base.exporter
import gws.lib.gdalx


gws.ext.new.exporter('shapefile')


class Config(gws.base.exporter.Config):
    """Shapefile Exporter configuration."""

    pass


class Props(gws.base.exporter.Props):
    """Shapefile Exporter properties."""

    pass


class Object(gws.base.exporter.Object):
    supportsVector = True
    supportsRaster = False

    def configure(self):
        self.withSplitLayers = True

    def run(self, ea, er):
        gws.base.exporter.util.run_gdal_vector_export('ESRI Shapefile', '', ea, er)
