"""Exporter for the GML format.

See https://gdal.org/en/stable/drivers/vector/gml.html#dataset-creation-options for supported options.
"""

import gws
import gws.base.exporter
import gws.lib.mime


gws.ext.new.exporter('gml')

class Config(gws.base.exporter.Config):
    """GML Exporter configuration."""

    pass


class Props(gws.base.exporter.Props):
    """GML Exporter properties."""

    pass


class Object(gws.base.exporter.Object):
    supportsVector = True
    supportsRaster = False
    supportsMultiLayer = True

    def run(self, ea, er):
        gws.base.exporter.util.run_gdal_vector_export('GML', gws.lib.mime.GML, ea, er)
