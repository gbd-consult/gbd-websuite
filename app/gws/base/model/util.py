"""Model utilities."""

import datetime

import gws
import gws.types as t

_ATTR_TO_PY = {
    gws.AttributeType.bool: bool,
    gws.AttributeType.bytes: bytes,
    gws.AttributeType.date: datetime.date,
    gws.AttributeType.datetime: datetime.datetime,
    # gws.AttributeType.feature
    # gws.AttributeType.featurelist
    gws.AttributeType.float: float,
    # gws.AttributeType.floatlist
    # gws.AttributeType.geometry
    gws.AttributeType.int: int,
    # gws.AttributeType.intlist
    gws.AttributeType.str: str,
    # gws.AttributeType.strlist
    gws.AttributeType.time: datetime.time,
}


def describe_from_feature_data(fd: gws.FeatureRecord) -> gws.DataSetDescription:
    py_to_attr = {str(v): k for k, v in _ATTR_TO_PY.items()}

    desc = gws.DataSetDescription(columns=[])

    for k, v in fd.attributes:
        typ = str(type(v))
        desc.columns.append(gws.ColumnDescription(
            name=k,
            nativeType=typ,
            type=py_to_attr.get(typ, gws.AttributeType.str),
        ))
        if fd.shape:
            col = gws.ColumnDescription(
                name='geometry',
                geometryType=fd.shape.type,
                geometrySrid=fd.shape.crs.srid,
                type=gws.AttributeType.geometry,
            )
            desc.columns.append(col)
            desc.geometryName = col.name
            desc.geometryType = col.geometryType
            desc.geometrySrid = col.geometrySrid

    desc.columnMap = {col.name: col for col in desc.columns}
    return desc


def clone_context(mc: gws.ModelContext, **kwargs) -> gws.ModelContext:
    return gws.ModelContext(gws.merge(mc, **kwargs))


def secondary_context(mc: gws.ModelContext, **kwargs) -> gws.ModelContext:
    mc2 = clone_context(mc, **kwargs)
    mc2.relDepth = mc.relDepth + 1
    return mc2
