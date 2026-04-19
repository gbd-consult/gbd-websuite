"""Exporter for the ESRI Shapefile format."""

from typing import Callable, Optional

import gws
import gws.base.exporter
import gws.gis.gdalx
import gws.lib.zipx
import gws.lib.osx
import gws.lib.mime


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

    def run(self, ea):
        return run_export(ea)


##


def run_export(ea: gws.ExportArgs) -> gws.ExportResult:
    """Run the export for multiple features, make a zip archive with a shapefile per model."""

    er = gws.ExportResult(
        path='',
        numFiles=0,
        numFeaturesTotal=0,
        numFeaturesExported=0,
        errors=[],
    )

    if not ea.features:
        return er

    by_model: dict[str, list[gws.Feature]] = {}
    for f in ea.features:
        by_model.setdefault(f.model.uid, []).append(f)

    base_dir = gws.u.ephemeral_dir(gws.u.random_string(64))

    for _, features in by_model.items():
        f = features[0]
        title = f.model.title
        if not title and hasattr(f.model, 'tableName'):
            title = getattr(f.model, 'tableName').split('.')[-1]
        if not title:
            title = f.model.uid
        path = base_dir + '/' + gws.u.to_uid(title) + '.shp'
        export_features(features, path, er, notify=ea.notify)
        er.numFiles += 1

    er.path = base_dir + '/export.zip'
    er.mime = gws.lib.mime.ZIP
    paths = gws.lib.osx.find_files(base_dir, deep=False)
    gws.lib.zipx.zip_to_path(er.path, *paths, flat=True)

    return er


_NOTIFY_STEP = 1000
_MAX_ERRORS = 100

def export_features(
    features: list[gws.Feature],
    target_path: str,
    er: gws.ExportResult,
    notify: Optional[Callable] = None,
):
    """Export features to a shapefile.
    
    Args:
        features: list of features to export (must have the same model)
        target_path: path to the output shapefile
        er: export result object to update with progress and errors
        notify: optional callback to notify about progress, called with 'feature' argument
    """
    
    cols = {}
    f = features[0]

    for fld in f.model.fields:
        if gws.gis.gdalx.is_attribute_type_supported(fld.attributeType):
            cols[fld.name] = fld.attributeType

    recs: list[gws.FeatureRecord] = []
    geom_type = None
    crs = None

    for n, f in enumerate(features, 1):
        er.numFeaturesTotal += 1

        if notify and n % _NOTIFY_STEP == 0:
            notify('feature')
        
        uid = f.uid()

        sh = f.shape()
        if not sh:
            if len(er.errors) < _MAX_ERRORS:
                er.errors.append(f'{uid}: has no shape')
            continue
        if geom_type and geom_type != sh.type:
            if len(er.errors) < _MAX_ERRORS:
                er.errors.append(f'{uid}: inconsistent geometry types: {geom_type} and {sh.type}')
            continue
        if crs and crs != sh.crs:
            if len(er.errors) < _MAX_ERRORS:
                er.errors.append(f'{uid}: inconsistent CRS: {crs} and {sh.crs}')
            continue

        if not geom_type:
            geom_type = sh.type
            crs = sh.crs

        rec = gws.FeatureRecord()
        rec.attributes = {c: f.get(c) for c in cols}
        rec.shape = sh
        recs.append(rec)
        er.numFeaturesExported += 1

    if not recs:
        return

    with gws.gis.gdalx.open_vector(target_path, 'w', driver='ESRI Shapefile') as ds:
        la = ds.create_layer('', cols, geom_type, crs)
        la.insert(recs)
