"""Helpers for exporters."""

from typing import Optional
import gws
import gws.lib.gdalx
import gws.lib.mime
import gws.lib.osx
import gws.lib.zipx


class Group(gws.Data):
    """Export-ready features of one model."""

    title: str
    """Group title: the model title, table name or model uid."""
    records: list[gws.FeatureRecord]
    """Records with the exported attributes and the shape."""
    columns: dict[str, gws.AttributeType]
    """Exported columns and their types."""
    geomType: Optional[gws.GeometryType]
    """Geometry type, taken from the first feature with a shape."""
    crs: Optional[gws.Crs]
    """CRS, taken from the first feature with a shape."""


def group_features(ea: gws.ExportArgs, er: gws.ExportResult) -> list[Group]:
    """Group features by model and determine the export columns and geometry type.

    Columns are the model fields with an attribute type supported by the
    exporter (or by GDAL, if the exporter does not restrict the types).
    Features that do not fit their group are skipped and reported in
    ``er.errors``, see ``gws.base.exporter``.

    Args:
        ea: Export arguments.
        er: Export result, used to report errors.

    Returns:
        A list of groups, one per model that has exportable features.
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
    """Create a group for the features of one model, or ``None`` if no feature fits."""
    grp = Group(
        records=[],
        title='',
        columns={},
        geomType=None,
        crs=None,
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

    return grp if grp.records else None


def _feature_to_record(f: gws.Feature, grp: Group, ea: gws.ExportArgs, er: gws.ExportResult) -> Optional[gws.FeatureRecord]:
    """Convert a feature to a record of the group, or report an error and return ``None``."""
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


def run_gdal_vector_export(driver_name: str, mime_type: str, ea: gws.ExportArgs, er: gws.ExportResult):
    """Export features with a GDAL vector driver.

    Writes one file per group, or a single file with one layer per group if
    the exporter has ``withMultiLayer``. Several files are zipped. Sets the
    path, mime type and counts in the export result. Does nothing if there
    are no features to export.

    Args:
        driver_name: GDAL driver name, e.g. ``GeoJSON``.
        mime_type: Mime type of the result file.
        ea: Export arguments.
        er: Export result, filled by this function.

    Raises:
        ``gws.Error``: If the driver is not supported.
    """

    di = gws.lib.gdalx.get_driver(driver_name)
    if not di:
        raise gws.Error(f'unsupported driver: {driver_name}')

    groups = group_features(ea, er)
    if not groups:
        return

    base_dir = gws.u.ephemeral_dir(gws.u.random_string(64))
    ext = di.extensions[0] if di.extensions else 'dat'

    if ea.exporter.withMultiLayer:
        path = base_dir + f'/export.{ext}'
        with gws.lib.gdalx.open_vector(path, 'w', driver=di.name, options=ea.exporter.options) as ds:
            for grp in groups:
                la = ds.create_layer(grp.title, grp.columns, grp.geomType, grp.crs)
                fids = la.insert(grp.records)
                er.numFeaturesExported += len(fids)
    else:
        for grp in groups:
            path = base_dir + f'/{gws.u.to_uid(grp.title)}.{ext}'
            with gws.lib.gdalx.open_vector(path, 'w', driver=di.name, options=ea.exporter.options) as ds:
                la = ds.create_layer(grp.title, grp.columns, grp.geomType, grp.crs)
                fids = la.insert(grp.records)
                er.numFeaturesExported += len(fids)

    paths = list(gws.lib.osx.find_files(base_dir, deep=False))
    er.numFiles = len(paths)
    if er.numFiles > 1:
        er.path = base_dir + '/export.zip'
        er.mimeType = gws.lib.mime.ZIP
        gws.lib.zipx.zip_to_path(er.path, paths, flat=True)
    else:
        er.path = paths[0]
        er.mimeType = mime_type
