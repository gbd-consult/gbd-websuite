import gws
import gws.types as t


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
    isRequired: t.Optional[bool]

    values: t.Optional[list[gws.ext.config.modelValue]]
    validators: t.Optional[list[gws.ext.config.modelValidator]]
    serverDefault: t.Optional[str]

    widget: t.Optional[gws.ext.config.modelWidget]


##

class Object(gws.Node, gws.IModelField):
    _validatorsIndex: dict[gws.Access, list[gws.IModelValidator]]
    _valuesIndex: dict[tuple[bool, gws.Access], gws.IModelValue]

    def configure(self):
        self.model = self.cfg('_defaultModel')
        self.name = self.cfg('name')
        self.title = self.cfg('title', default=self.name)
        self.isPrimaryKey = self.cfg('isPrimaryKey')
        self.isRequired = self.cfg('isRequired')

        self.values = []
        self.validators = []
        self.widget = None
        self.serverDefault = self.cfg('serverDefault')

        self.configure_values()
        self.configure_validators()
        self.configure_widget()

    def configure_values(self):
        p = self.cfg('values')
        if p:
            self.values = self.create_children(gws.ext.object.modelValue, p)
            return True

    def configure_validators(self):
        p = self.cfg('validators')
        if p:
            self.validators = self.create_children(gws.ext.object.modelValidator, p)
            return True

    def configure_widget(self):
        p = self.cfg('widget')
        if p:
            self.widget = self.create_child(gws.ext.object.modelWidget, p)
            return True

    def post_configure(self):
        self._create_validators_index()
        self._create_mandatory_validators()
        self._create_values_index()

    def _create_validators_index(self):
        self._validatorsIndex = {gws.Access.write: [], gws.Access.create: []}

        for vd in self.validators:
            if vd.forWrite:
                self._validatorsIndex[gws.Access.write].append(vd)
            if vd.forCreate:
                self._validatorsIndex[gws.Access.create].append(vd)

    def _create_mandatory_validators(self):
        vd = self.root.create_shared(
            gws.ext.object.modelValidator,
            type='format',
            uid='gws.base.model.field.default_validator_format')
        self._validatorsIndex[gws.Access.write].append(vd)
        self._validatorsIndex[gws.Access.create].append(vd)

        if not self.isRequired:
            return

        vd = self.root.create_shared(
            gws.ext.object.modelValidator,
            type='required',
            uid='gws.base.model.field.default_validator_required')
        self._validatorsIndex[gws.Access.write].append(vd)
        self._validatorsIndex[gws.Access.create].append(vd)

    def _create_values_index(self):
        self._valuesIndex = {}

        for va in self.values:
            if va.forRead:
                self._valuesIndex[va.isDefault, gws.Access.read] = va
            if va.forWrite:
                self._valuesIndex[va.isDefault, gws.Access.write] = va
            if va.forCreate:
                self._valuesIndex[va.isDefault, gws.Access.create] = va

    ##

    def props(self, user):
        wp = None
        if self.widget:
            wp = gws.props(self.widget, user)
            if not user.can_write(self):
                wp.readOnly = True

        return Props(
            attributeType=self.attributeType,
            name=self.name,
            title=self.title,
            type=self.extType,
            widget=wp,
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

    ##

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
                props.attributes[self.name] = val

    ##

    def compute(self, feature, access, user, **kwargs):
        val = self._valuesIndex.get((False, access))
        if val:
            feature.attributes[self.name] = val.compute(feature, self, user, **kwargs)
            return
        if self.name in feature.attributes:
            return
        val = self._valuesIndex.get((True, access))
        if val:
            feature.attributes[self.name] = val.compute(feature, self, user, **kwargs)
            return

    def validate(self, feature, access, user, **kwargs):
        for vd in self._validatorsIndex[access]:
            ok = vd.validate(feature, self, user, **kwargs)
            if not ok:
                feature.errors.append(gws.ModelValidationError(
                    fieldName=self.name,
                    message=vd.message,
                ))
                return False

        return True

    ##

    def columns(self):
        return []

    def orm_properties(self):
        return {}
