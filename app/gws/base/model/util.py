"""Model utilities."""

from typing import Iterable
import datetime

import gws
import gws.base.feature

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


def iter_features(features: list[gws.Feature], mc: gws.ModelContext) -> Iterable[gws.Feature]:
    for f in features:
        yield f

        if mc.relDepth >= mc.maxDepth:
            continue

        sub = []

        for val in f.attributes.values():
            if isinstance(val, gws.base.feature.Feature):
                sub.append(val)
            elif isinstance(val, list):
                for v in val:
                    if isinstance(v, gws.base.feature.Feature):
                        sub.append(val)

        yield from iter_features(sub, context_from(mc))


def describe_from_record(fd: gws.FeatureRecord) -> gws.DataSetDescription:
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


def context_from(mc: gws.ModelContext, **kwargs) -> gws.ModelContext:
    return gws.ModelContext(gws.u.merge(mc, kwargs, relDepth=mc.relDepth + 1))
