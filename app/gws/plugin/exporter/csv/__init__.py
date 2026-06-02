"""Exporter for the CSV format.

See https://gdal.org/en/stable/drivers/vector/gml.html#dataset-creation-options for supported options.
"""

import gws
import gws.base.exporter
import gws.lib.mime


gws.ext.new.exporter('csv')

class Config(gws.base.exporter.Config):
    """CSV Exporter configuration."""

    pass


class Props(gws.base.exporter.Props):
    """CSV Exporter properties."""

    pass


class Object(gws.base.exporter.Object):
    supportsVector = True
    supportsRaster = False

    def configure(self):
        self.withSplitLayers = True
        self.withNoGeometry = True

    def run(self, ea, er):
        gws.base.exporter.util.run_gdal_vector_export('CSV', gws.lib.mime.CSV, ea, er)
