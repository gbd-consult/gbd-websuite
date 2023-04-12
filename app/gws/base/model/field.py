import gws
import gws.base.database.model
import gws.lib.sa as sa

import gws.types as t


class ErrorValue:
    pass


class Relation(gws.Data):
    modelUid: str
    fieldName: str = ''
    discriminator: str = ''
    title: str = ''


class Props(gws.Props):
    attributeType: gws.AttributeType
    geometryType: gws.GeometryType
    name: str
    relations: list[Relation]
    title: str
    type: str
    widget: gws.ext.props.modelWidget
    uid: str


class Config(gws.ConfigWithAccess):
    name: str
    title: t.Optional[str]
    isPrimaryKey: bool = False

    values: t.Optional[list[gws.ext.config.modelValue]]
    validators: t.Optional[list[gws.ext.config.modelValidator]]

    widget: t.Optional[gws.ext.config.modelWidget]
    #
    # relation: t.Optional[FieldRelationConfig]
    # relations: t.Optional[list[FieldRelationConfig]]
    #
    # foreignKey: t.Optional[FieldNameTypeConfig]
    # discriminatorKey: t.Optional[FieldNameTypeConfig]
    # link: t.Optional[FieldLinkConfig]
    #
    # filePath: t.Optional[str]
    #
    # value: t.Optional[FieldValueConfig]
    # validators: t.Optional[list[ValidatorConfig]]
    #
    # geometryType: t.Optional[str]
    #
    # errorMessages: t.Optional[dict]

    # permissions: t.Optional[FieldPermissions]
    # textSearch: t.Optional[FieldTextSearchConfig]


##

class Object(gws.Node, gws.IModelField):

    def configure(self):
        p = self.permissions.get(gws.Access.use)
        if p:
            self.permissions.setdefault(gws.Access.read, p)

        self.model = self.cfg('_model')
        self.name = self.cfg('name')
        self.title = self.cfg('title', default=self.name)
        self.isPrimaryKey = self.cfg('isPrimaryKey')

        self.configure_values()
        self.configure_validators()
        self.configure_widget()

    def configure_values(self):

        self.fixedValues = {}
        self.defaultValues = {}

        for cfg in self.cfg('values', default=[]):
            v = self.root.create(gws.ext.object.modelValue, config=cfg)
            d = self.defaultValues if v.isDefault else self.fixedValues
            if v.forRead:
                d[gws.Access.read] = v
            if v.forWrite:
                d[gws.Access.write] = v
            if v.forCreate:
                d[gws.Access.create] = v

    def configure_validators(self):

        self.validators = {
            gws.Access.write: [],
            gws.Access.create: [],
        }

        for cfg in self.cfg('validators', default=[]):
            v = self.root.create(gws.ext.object.modelValidator, config=cfg)
            if v.forWrite:
                self.validators[gws.Access.write].append(v)
            if v.forCreate:
                self.validators[gws.Access.create].append(v)

    def configure_widget(self):
        p = self.cfg('widget')
        if p:
            self.widget = self.create_child(gws.ext.object.modelWidget, p)
            return True

    ##

    def props(self, user):
        return Props(
            attributeType=self.attributeType,
            name=self.name,
            title=self.title,
            type=self.extType,
            widget=self.widget,
            uid=self.uid,
        )

    ##

    def db_to_py(self, val):
        return val

    def prop_to_py(self, val):
        return val

    def py_to_db(self, val):
        return val

    def py_to_prop(self, val):
        return val

    def load_from_record(self, feature, record, user, relation_depth=0, **kwargs):
        if hasattr(record, self.name):
            val = getattr(record, self.name)
            if val is not None:
                val = self.db_to_py(val)
            if val is not None:
                feature.attributes[self.name] = val

    def load_from_data(self, feature, data, user, relation_depth=0, **kwargs):
        val = data.attributes.get(self.name)
        if val is not None:
            val = self.prop_to_py(val)
        if val is not None:
            feature.attributes[self.name] = val

    def load_from_props(self, feature, props, user, relation_depth=0, **kwargs):
        val = props.attributes.get(self.name)
        if val is not None:
            val = self.prop_to_py(val)
        if val is not None:
            feature.attributes[self.name] = val

    def _value_to_write(self, feature: gws.IFeature):
        val = feature.attributes.get(self.name)
        if val is None:
            return False, None
        if val is gws.ErrorValue:
            raise gws.Error(f'attempt to store an error value, field {feature.model.uid!r}.{self.name!r}, feature {feature.uid()!r}')
        return True, val

    def store_to_record(self, feature, record, user, **kwargs):
        ok, val = self._value_to_write(feature)
        if ok:
            val = self.py_to_db(val)
            if val is not None:
                setattr(record, self.name, val)

    def store_to_props(self, feature, props, user, **kwargs):
        ok, val = self._value_to_write(feature)
        if ok:
            val = self.py_to_prop(val)
            if val is not None:
                #     return
                # if isinstance(val, gws.Object):
                #     val = gws.props(val, user, self)
                props.attributes[self.name] = val

    def compute(self, feature, access, user, **kwargs):
        val = self.fixedValues.get(access)
        if val:
            feature.attributes[self.name] = val.compute(feature, self, user, **kwargs)
            return
        if self.name in feature.attributes:
            return
        val = self.defaultValues.get(access)
        if val:
            feature.attributes[self.name] = val.compute(feature, self, user, **kwargs)
            return

    def validate(self, feature, access, user, **kwargs):
        for v in self.validators[access]:
            ok = v.validate(feature, self, user, **kwargs)
            if not ok:
                feature.errors.append(gws.ModelValidationError(
                    fieldName=self.name,
                    message=v.message,
                ))
                return False
        return True

    ##

    def columns(self):
        return []

    def orm_properties(self):
        return {}


##

_SCALAR_TYPES = {
    gws.AttributeType.bool: sa.Boolean,
    gws.AttributeType.date: sa.Date,
    gws.AttributeType.datetime: sa.DateTime,
    gws.AttributeType.float: sa.Float,
    gws.AttributeType.int: sa.Integer,
    gws.AttributeType.str: sa.String,
    gws.AttributeType.time: sa.Time,
    gws.AttributeType.geometry: sa.geo.Geometry,
}


class Scalar(Object):
    def columns(self):
        kwargs = {}
        if self.isPrimaryKey:
            kwargs['primary_key'] = True
        # if self.value.serverDefault:
        #     kwargs['server_default'] = sa.text(self.value.serverDefault)
        col = sa.Column(self.name, _SCALAR_TYPES[self.attributeType], **kwargs)
        return [col]


##


##


# @TODO this should be populated dynamically from available gws.ext.object.modelField types

_DEFAULT_FIELD_TYPES = {
    gws.AttributeType.str: 'text',
    gws.AttributeType.int: 'integer',
    gws.AttributeType.date: 'date',

    # gws.AttributeType.bool: 'bool',
    # gws.AttributeType.bytes: 'bytes',
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


def config_from_column(column: gws.ColumnDescription) -> gws.Config:
    typ = _DEFAULT_FIELD_TYPES.get(column.type)
    if not typ:
        raise gws.Error(f'cannot find suitable field type for column {column.name!r}:{column.type!r}')
    return gws.Config(
        type=typ,
        name=column.name,
        isPrimaryKey=column.isPrimaryKey,
    )
