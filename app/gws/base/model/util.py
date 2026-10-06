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
    """Iterate over features and their related features.

    Related features are features stored in attributes, directly or in lists. They are
    visited depth-first, as long as the relation depth of the context is below its maximum.

    Args:
        features: The features.
        mc: The model context.

    Yields:
        Each feature, followed by its related features.
    """
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
                        sub.append(v)

        yield from iter_features(sub, secondary_context(mc))


def describe_from_record(fd: gws.FeatureRecord) -> gws.DataSetDescription:
    """Create a dataset description from a feature record.

    Column types are derived from the python types of the attribute values. If the record
    has a shape, a ``geometry`` column is added.

    Args:
        fd: The feature record.

    Returns:
        The dataset description.
    """
    py_to_attr = {str(v): k for k, v in _ATTR_TO_PY.items()}

    desc = gws.DataSetDescription(columns=[])

    for k, v in fd.attributes.items():
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


def copy_context(mc: gws.ModelContext, **kwargs) -> gws.ModelContext:
    """Copy a model context.

    Args:
        mc: The model context.
        **kwargs: Properties to set in the copy.

    Returns:
        A new model context.
    """
    return gws.ModelContext(gws.u.merge(mc, kwargs))

def secondary_context(mc: gws.ModelContext, **kwargs) -> gws.ModelContext:
    """Copy a model context for related features, with the relation depth increased by one.

    Args:
        mc: The model context.
        **kwargs: Properties to set in the copy.

    Returns:
        A new model context.
    """
    return gws.ModelContext(gws.u.merge(mc, kwargs, relDepth=mc.relDepth + 1))
