"""Exporter for the ESRI Shapefile format."""

import gws
import gws.base.exporter
import gws.lib.gdalx


@gws.ext.config.exporter('shapefile')
class Config(gws.base.exporter.Config):
    """Shapefile Exporter configuration."""

    pass


@gws.ext.props.exporter('shapefile')
class Props(gws.base.exporter.Props):
    """Shapefile Exporter properties."""

    pass


@gws.ext.object.exporter('shapefile')
class Object(gws.base.exporter.Object):
    supportsVector = True
    supportsRaster = False
    supportsMultiLayer = False

    def run(self, ea, er):
        gws.base.exporter.util.run_gdal_vector_export('ESRI Shapefile', '', ea, er)
