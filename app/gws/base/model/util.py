"""Model utilities."""

import datetime

import gws
import gws.types as t


def locate(
        models: list[gws.IModel],
        user: gws.IUser = None,
        access: gws.Access = None,
        uid: str = None
) -> t.Optional[gws.IModel]:
    for model in models:
        if user and access and not user.can(access, model):
            continue
        if uid and model.uid != uid:
            continue
        return model


# @TODO this should be populated dynamically from available gws.ext.object.modelField types

_DEFAULT_FIELD_TYPES = {
    gws.AttributeType.str: 'text',
    gws.AttributeType.int: 'integer',
    gws.AttributeType.date: 'date',

    # gws.AttributeType.bool: 'bool',
    # gws.AttributeType.bytes: 'bytea',
    # gws.AttributeType.datetime: 'datetime',
    # gws.AttributeType.feature: 'feature',
    # gws.AttributeType.featurelist: 'featurelist',
    gws.AttributeType.float: 'float',
    # gws.AttributeType.floatlist: 'floatlist',
    gws.AttributeType.geometry: 'geometry',
    # gws.AttributeType.intlist: 'intlist',
    # gws.AttributeType.strlist: 'strlist',
    # gws.AttributeType.time: 'time',
}

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


def describe_from_feature_data(fd: gws.FeatureData) -> gws.DataSetDescription:
    py_to_attr = {str(v): k for k, v in _ATTR_TO_PY.items()}

    desc = gws.DataSetDescription(columns=[])

    for k, v in fd.attributes:
        typ = str(type(v))
        desc.columns[k] = gws.ColumnDescription(
            name=k,
            nativeType=typ,
            type=py_to_attr.get(typ, gws.AttributeType.str),
        )
        if fd.shape:
            col = gws.ColumnDescription(
                name='geometry',
                geometryType=fd.shape.type,
                geometrySrid=fd.shape.crs.srid,
                type=gws.AttributeType.geometry,
            )
            desc.columns[col.name] = col
            desc.geometryName = col.name
            desc.geometryType = col.geometryType
            desc.geometrySrid = col.geometrySrid

    return desc


def field_config_from_column(column: gws.ColumnDescription) -> t.Optional[gws.Config]:
    typ = _DEFAULT_FIELD_TYPES.get(column.type)
    if not typ:
        gws.log.warning(f'cannot find suitable field type for column {column.name!r}:{column.type!r}')
        return
    return gws.Config(
        type=typ,
        name=column.name,
        isPrimaryKey=column.isPrimaryKey,
        isRequired=not column.isNullable,
    )
