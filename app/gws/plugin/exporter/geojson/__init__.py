"""Exporter for the GeoJSON format.

See https://gdal.org/en/stable/drivers/vector/geojson.html#layer-creation-options for supported options.
"""

import gws
import gws.base.exporter
import gws.lib.mime


gws.ext.new.exporter('geojson')

class Config(gws.base.exporter.Config):
    """GeoJSON Exporter configuration."""

    pass


class Props(gws.base.exporter.Props):
    """GeoJSON Exporter properties."""

    pass


class Object(gws.base.exporter.Object):
    supportsVector = True
    supportsRaster = False
    supportsMultiLayer = False

    def run(self, ea, er):
        gws.base.exporter.util.run_gdal_vector_export('GeoJSON', gws.lib.mime.GEOJSON, ea, er)
