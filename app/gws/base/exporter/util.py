"""Export utilities."""

from typing import Optional
import gws
import gws.lib.gdalx
import gws.lib.mime
import gws.lib.osx
import gws.lib.zipx


class Group(gws.Data):
    title: str
    records: list[gws.FeatureRecord]
    columns: dict[str, gws.AttributeType]
    geomType: Optional[gws.GeometryType]
    crs: Optional[gws.Crs]


def group_features(ea: gws.ExportArgs, er: gws.ExportResult) -> list[Group]:
    """Group features by model, determine export columns and geometry type.

    Args:
        ea: Export arguments.
        er: Export result, used to report errors and counts.

    Returns:
        List of groups, one per model, with export-ready records.
    """

    if not ea.features:
        return []

    by_model: dict[str, list[gws.Feature]] = {}
    for f in ea.features:
        by_model.setdefault(f.model.uid, []).append(f)

    ls = []

    for _, features in by_model.items():
        grp = _create_group(features, ea, er)
        if grp:
            ls.append(grp)

    return ls


def _create_group(features: list[gws.Feature], ea: gws.ExportArgs, er: gws.ExportResult) -> Optional[Group]:
    grp = Group(
        records=[],
        title='',
        columns={},
        geomType=None,
        srid=0,
    )

    f = features[0]

    grp.title = f.model.title
    if not grp.title and hasattr(f.model, 'tableName'):
        grp.title = getattr(f.model, 'tableName').split('.')[-1]
    if not grp.title:
        grp.title = f.model.uid

    types = ea.exporter.supportedAttributeTypes or gws.lib.gdalx.supported_attribute_types()
    for fld in f.model.fields:
        if fld.attributeType in types:
            grp.columns[fld.name] = fld.attributeType

    for n, f in enumerate(features, 1):
        if ea.notify:
            ea.notify('')
        rec = _feature_to_record(f, grp, ea, er)
        if rec:
            grp.records.append(rec)
            er.numFeaturesExported += 1

    return grp if grp.records else None


def _feature_to_record(f: gws.Feature, grp: Group, ea: gws.ExportArgs, er: gws.ExportResult) -> Optional[gws.FeatureRecord]:
    uid = f.uid()

    sh = f.shape()

    if sh:
        if grp.geomType and grp.geomType != sh.type and not ea.exporter.withMixedGeometry:
            if len(er.errors) < ea.maxErrors:
                er.errors.append(f'{uid}: inconsistent geometry: {grp.geomType} and {sh.type}')
            return

        if grp.crs and grp.crs != sh.crs and not ea.exporter.withMixedCrs:
            if len(er.errors) < ea.maxErrors:
                er.errors.append(f'{uid}: inconsistent CRS: {grp.crs} and {sh.crs}')
            return

        if not grp.geomType:
            grp.geomType = sh.type
            grp.crs = sh.crs

    elif not ea.exporter.withNoGeometry:
        if len(er.errors) < ea.maxErrors:
            er.errors.append(f'{uid}: has no geometry')
        return

    return gws.FeatureRecord(
        attributes={c: f.get(c) for c in grp.columns},
        shape=sh,
    )


##


def zip_all(base_dir: str, er: gws.ExportResult):
    er.path = base_dir + '/export.zip'
    er.mime = gws.lib.mime.ZIP
    paths = gws.lib.osx.find_files(base_dir, deep=False)
    n = gws.lib.zipx.zip_to_path(er.path, list(paths), flat=True)
    er.numFiles += n


##


def run_gdal_vector_export(driver_name: str, mime: str, ea: gws.ExportArgs, er: gws.ExportResult):
    """Run the export for a GDAL vector driver."""

    di = gws.lib.gdalx.get_driver(driver_name)
    if not di:
        raise gws.Error(f'unsupported driver: {driver_name}')

    groups = group_features(ea, er)
    if not groups:
        return

    base_dir = gws.u.ephemeral_dir(gws.u.random_string(64))
    ext = di.extensions[0] if di.extensions else 'dat'

    if ea.exporter.withSplitTypes:
        for grp in groups:
            er.path = base_dir + f'/{gws.u.to_uid(grp.title)}.{ext}'
            with gws.lib.gdalx.open_vector(er.path, 'w', driver=di.name, options=ea.exporter.options) as ds:
                er.numFiles += 1
                la = ds.create_layer(grp.title, grp.columns, grp.geomType, grp.crs)
                la.insert(grp.records)
        zip_all(base_dir, er)
    else:
        er.path = base_dir + f'/export.{ext}'
        er.mime = mime
        with gws.lib.gdalx.open_vector(er.path, 'w', driver=di.name, options=ea.exporter.options) as ds:
            er.numFiles += 1
            for grp in groups:
                la = ds.create_layer(grp.title, grp.columns, grp.geomType, grp.crs)
                la.insert(grp.records)
