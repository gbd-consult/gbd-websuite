"""KML exporter.

Exports vector features with the GDAL ``KML`` driver. With
``withMultiLayer``, all models are written to one file. GDAL creation
options can be passed with ``options``, see
https://gdal.org/en/stable/drivers/vector/kml.html.

Example::

    exporters+ {
        type "kml"
        title "KML"
        target "download"
    }
"""

import gws
import gws.base.exporter
import gws.lib.mime


@gws.ext.config.exporter('kml')
class Config(gws.base.exporter.Config):
    """Export of features to KML."""

    pass


@gws.ext.props.exporter('kml')
class Props(gws.base.exporter.Props):
    """KML Exporter properties."""

    pass


@gws.ext.object.exporter('kml')
class Object(gws.base.exporter.Object):
    """KML exporter."""

    supportsVector = True
    supportsRaster = False
    supportsMultiLayer = True

    def run(self, ea, er):
        gws.base.exporter.util.run_gdal_vector_export('KML', gws.lib.mime.KML, ea, er)
