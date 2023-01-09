import gws
import gws.types as t

from . import value


class Props(gws.Props):
    name: str
    type: str
    title: str
    attributeType: gws.AttributeType
    geometryType: t.Optional[gws.GeometryType]


class Config(gws.ConfigWithAccess):
    name: str
    title: t.Optional[str]
    isRequired: bool = False
    isUnique: bool = False
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
    def configure(self):
        p = self.permissions.get(gws.Access.use)
        if p:
            self.permissions.setdefault(gws.Access.read, p)

        self.model = self.var('_model')
        self.name = self.var('name')
        self.title = self.var('title', default=self.name)

        self.isPrimaryKey = self.var('isPrimaryKey')
        self.isRequired = self.var('isRequired')
        self.isUnique = self.var('isUnique')

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

    def load_from_dict(self, feature, attributes, user, **kwargs):
        if self.name in attributes:
            feature.attributes[self.name] = attributes[self.name]

    def load_from_record(self, feature, record, user, **kwargs):
        if hasattr(record, self.name):
            feature.attributes[self.name] = getattr(record, self.name)

    def store_to_dict(self, feature, attributes, user, **kwargs):
        if self.name in feature.attributes:
            attributes[self.name] = feature.attributes[self.name]

    def store_to_record(self, feature, record, user, **kwargs):
        if self.name in feature.attributes:
            setattr(record, self.name, feature.attributes[self.name])

    def compute_value(self, feature, access, user, **kwargs):
        v = self.fixedValues.get(access)
        if v:
            feature.attributes[self.name] = v.compute(feature, self, user, **kwargs)
            return
        if self.name in feature.attributes:
            feature.attributes[self.name] = self.convert_value(feature, access, user)
            return
        v = self.defaultValues.get(access)
        if v:
            feature.attributes[self.name] = v.compute(feature, self, user, **kwargs)
            return

    def convert_value(self, feature, access, user, **kwargs):
        return feature.attributes.get(self.name)

    def validate_value(self, feature, access, user, **kwargs):
        for v in self.validators[access]:
            ok = v.validate(feature, self, user, **kwargs)
            if not ok:
                feature.errors.append(gws.ModelValidationError(
                    fieldName=self.name,
                    message=v.message
                ))
                return False
        return True
