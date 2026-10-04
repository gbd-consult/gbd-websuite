"""GeoJSON exporter.

Exports vector features with the GDAL ``GeoJSON`` driver. Each model is
written to its own file. GDAL creation options can be passed with
``options``, see
https://gdal.org/en/stable/drivers/vector/geojson.html.

Example::

    exporters+ {
        type "geojson"
        title "GeoJSON"
        target "download"
    }
"""

import gws
import gws.base.exporter
import gws.lib.mime


@gws.ext.config.exporter('geojson')
class Config(gws.base.exporter.Config):
    """Export of features to GeoJSON."""

    pass


@gws.ext.props.exporter('geojson')
class Props(gws.base.exporter.Props):
    """GeoJSON Exporter properties."""

    pass


@gws.ext.object.exporter('geojson')
class Object(gws.base.exporter.Object):
    """GeoJSON exporter."""

    supportsVector = True
    supportsRaster = False
    supportsMultiLayer = False

    def run(self, ea, er):
        gws.base.exporter.util.run_gdal_vector_export('GeoJSON', gws.lib.mime.GEOJSON, ea, er)
