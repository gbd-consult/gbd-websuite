"""ESRI Shapefile exporter.

Exports vector features with the GDAL ``ESRI Shapefile`` driver. Each model is
written to its own file. GDAL creation options can be passed with
``options``, see https://gdal.org/en/stable/drivers/vector/shapefile.html.

Example::

    exporters+ {
        type "shapefile"
        title "Shapefile"
        target "download"
    }
"""

import gws
import gws.base.exporter
import gws.lib.gdalx


@gws.ext.config.exporter('shapefile')
class Config(gws.base.exporter.Config):
    """Export of features to ESRI Shapefile."""

    pass


@gws.ext.props.exporter('shapefile')
class Props(gws.base.exporter.Props):
    """Shapefile Exporter properties."""

    pass


@gws.ext.object.exporter('shapefile')
class Object(gws.base.exporter.Object):
    """ESRI Shapefile exporter."""

    supportsVector = True
    supportsRaster = False
    supportsMultiLayer = False

    def run(self, ea, er):
        gws.base.exporter.util.run_gdal_vector_export('ESRI Shapefile', '', ea, er)
