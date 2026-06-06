"""Exporter for the KML format.

See https://gdal.org/en/stable/drivers/vector/kml.html#creation-options for supported options.
"""

import gws
import gws.base.exporter
import gws.lib.mime


gws.ext.new.exporter('kml')


class Config(gws.base.exporter.Config):
    """KML Exporter configuration."""

    pass


class Props(gws.base.exporter.Props):
    """KML Exporter properties."""

    pass


class Object(gws.base.exporter.Object):
    supportsVector = True
    supportsRaster = False
    supportsMultiLayer = True

    def run(self, ea, er):
        gws.base.exporter.util.run_gdal_vector_export('KML', gws.lib.mime.KML, ea, er)
