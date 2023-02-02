import gws
import gws.types as t


class ErrorValue:
    pass


class Props(gws.Props):
    attributeType: gws.AttributeType
    geometryType: t.Optional[gws.GeometryType]
    name: str
    title: str
    type: str
    widget: gws.ext.props.modelWidget


class Config(gws.ConfigWithAccess):
    name: str
    title: t.Optional[str]
    isPrimaryKey: bool = False

    values: t.Optional[t.List[gws.ext.config.modelValue]]
    validators: t.Optional[t.List[gws.ext.config.modelValidator]]

    # widget: t.Optional[WidgetConfig]
    #
    # relation: t.Optional[FieldRelationConfig]
    # relations: t.Optional[t.List[FieldRelationConfig]]
    #
    # foreignKey: t.Optional[FieldNameTypeConfig]
    # discriminatorKey: t.Optional[FieldNameTypeConfig]
    # link: t.Optional[FieldLinkConfig]
    #
    # filePath: t.Optional[str]
    #
    # value: t.Optional[FieldValueConfig]
    # validators: t.Optional[t.List[ValidatorConfig]]
    #
    # geometryType: t.Optional[str]
    #
    # errorMessages: t.Optional[dict]

    # permissions: t.Optional[FieldPermissions]
    # textSearch: t.Optional[FieldTextSearchConfig]


class Object(gws.Node, gws.IModelField):
    widget: t.Optional[gws.IModelWidget] = None

    fixedValues: t.Dict[gws.Access, gws.IModelValue]
    defaultValues: t.Dict[gws.Access, gws.IModelValue]

    validators: t.Dict[gws.Access, t.List[gws.IModelValidator]]

    geometryType = None
    geometryCrs = None

    def configure(self):
        p = self.permissions.get(gws.Access.use)
        if p:
            self.permissions.setdefault(gws.Access.read, p)

        self.model = self.var('_model')
        self.name = self.var('name')
        self.title = self.var('title', default=self.name)
        self.isPrimaryKey = self.var('isPrimaryKey')

        self.configure_values()
        self.configure_validators()

    def configure_values(self):

        self.fixedValues = {}
        self.defaultValues = {}

        for cfg in self.var('values', default=[]):
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

        for cfg in self.var('validators', default=[]):
            v = self.root.create(gws.ext.object.modelValidator, config=cfg)
            if v.forWrite:
                self.validators[gws.Access.write].append(v)
            if v.forCreate:
                self.validators[gws.Access.create].append(v)

    ##

    def props(self, user):
        return Props(
            attributeType=self.attributeType,
            geometryType=self.geometryType,
            name=self.name,
            title=self.title,
            type=self.extType,
            widget=self.widget
        )

    def load_from_record(self, feature, record, user, **kwargs):
        if hasattr(record, self.name):
            val = getattr(record, self.name)
            feature.attributes[self.name] = val

    def load_from_data(self, feature, data, user, **kwargs):
        if self.name in data.attributes:
            val = data.attributes[self.name]
            feature.attributes[self.name] = val

    def load_from_props(self, feature, props, user, **kwargs):
        if self.name in props.attributes:
            val = props.attributes[self.name]
            feature.attributes[self.name] = val

    def value_to_store(self, feature: gws.IFeature):
        if self.name not in feature.attributes:
            return False, None
        val = feature.attributes[self.name]
        if val == gws.ErrorValue:
            raise gws.Error(f'attempt to store an error value for {self.name!r}')
        return True, val

    def store_to_record(self, feature, record, user, **kwargs):
        ok, val = self.value_to_store(feature)
        if ok:
            setattr(record, self.name, val)

    def store_to_data(self, feature, data, user, **kwargs):
        ok, val = self.value_to_store(feature)
        if ok:
            data.attributes[self.name] = val

    def store_to_props(self, feature, props, user, **kwargs):
        ok, val = self.value_to_store(feature)
        if ok:
            if isinstance(val, gws.Object):
                val = gws.props(val, user, self)
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
                    message=v.message
                ))
                return False
        return True


##

_DEFAULT_FIELD_TYPES = {
    gws.AttributeType.str: 'text',
    gws.AttributeType.int: 'integer',

    # gws.AttributeType.bool: 'bool',
    # gws.AttributeType.bytes: 'bytes',
    # gws.AttributeType.date: 'date',
    # gws.AttributeType.datetime: 'datetime',
    # gws.AttributeType.feature: 'feature',
    # gws.AttributeType.featurelist: 'featurelist',
    # gws.AttributeType.float: 'float',
    # gws.AttributeType.floatlist: 'floatlist',
    # gws.AttributeType.geometry: 'geometry',
    # gws.AttributeType.intlist: 'intlist',
    # gws.AttributeType.strlist: 'strlist',
    # gws.AttributeType.time: 'time',
}


def config_from_column(column: gws.ColumnDescription) -> gws.Config:
    return gws.Config(
        type=_DEFAULT_FIELD_TYPES[column.type]
    )
