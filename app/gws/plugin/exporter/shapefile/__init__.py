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

    def run(self, ea, er):
        return run_export(ea, er)


##


def run_export(ea: gws.ExportArgs, er: gws.ExportResult):
    """Run the export for multiple features, make a zip archive with a shapefile per model."""

    groups = gws.base.exporter.util.group_features(ea, er)
    if not groups:
        return

    base_dir = gws.u.ephemeral_dir(gws.u.random_string(64))
    for grp in groups:
        path = base_dir + '/' + gws.u.to_uid(grp.title) + '.shp'
        with gws.lib.gdalx.open_vector(path, 'w', driver='ESRI Shapefile') as ds:
            la = ds.create_layer(grp.title, grp.columns, grp.geomType, grp.crs)
            la.insert(grp.records)

    gws.base.exporter.util.zip_all(base_dir, er)
