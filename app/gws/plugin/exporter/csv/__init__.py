"""CSV exporter.

Exports vector features with the GDAL ``CSV`` driver. Each model is
written to its own file. GDAL creation options can be passed with
``options``, see
https://gdal.org/en/stable/drivers/vector/csv.html.

Example::

    exporters+ {
        type "csv"
        title "CSV"
        target "download"
    }
"""

import gws
import gws.base.exporter
import gws.lib.mime


@gws.ext.config.exporter('csv')
class Config(gws.base.exporter.Config):
    """Export of features to CSV."""

    pass


@gws.ext.props.exporter('csv')
class Props(gws.base.exporter.Props):
    """CSV Exporter properties."""

    pass


@gws.ext.object.exporter('csv')
class Object(gws.base.exporter.Object):
    """CSV exporter."""

    supportsVector = True
    supportsRaster = False
    supportsMultiLayer = False

    def run(self, ea, er):
        gws.base.exporter.util.run_gdal_vector_export('CSV', gws.lib.mime.CSV, ea, er)
