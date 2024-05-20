from typing import Optional, cast

import gws


class Props(gws.Props):
    attributeType: gws.AttributeType
    geometryType: gws.GeometryType
    name: str
    title: str
    type: str
    widget: gws.ext.props.modelWidget
    uid: str
    relatedModelUids: list[str]


class Config(gws.ConfigWithAccess):
    name: str
    title: Optional[str]

    isPrimaryKey: Optional[bool]
    isRequired: Optional[bool]
    isUnique: Optional[bool]
    isAuto: Optional[bool]

    values: Optional[list[gws.ext.config.modelValue]]
    validators: Optional[list[gws.ext.config.modelValidator]]

    widget: Optional[gws.ext.config.modelWidget]


##

class Object(gws.ModelField):

    def configure(self):
        self.model = self.cfg('_defaultModel')
        self.name = self.cfg('name')
        self.title = self.cfg('title', default=self.name)

        self.values = []
        self.validators = []
        self.widget = None

        self.configure_flags()
        self.configure_values()
        self.configure_validators()
        self.configure_widget()

    def configure_flags(self):
        col = self.describe()

        p = self.cfg('isPrimaryKey')
        if p is not None:
            self.isPrimaryKey = p
        else:
            self.isPrimaryKey = col and col.isPrimaryKey

        p = self.cfg('isRequired')
        if p is not None:
            self.isRequired = p
        else:
            self.isRequired = col and not col.isNullable

        p = self.cfg('isAuto')
        if p is not None:
            self.isAuto = p
        else:
            self.isAuto = col and col.isAutoincrement

        p = self.cfg('isUnique')
        if p is not None:
            self.isUnique = p
        else:
            self.isUnique = False

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
        if self.isRequired:
            vd = self.root.create_shared(
                gws.ext.object.modelValidator,
                type='required',
                uid='gws.base.model.field.default_validator_required',
                forCreate=True,
                forUpdate=True,
            )
            self.validators.append(vd)

        vd = self.root.create_shared(
            gws.ext.object.modelValidator,
            type='format',
            uid='gws.base.model.field.default_validator_format',
            forCreate=True,
            forUpdate=True,
        )

        self.validators.append(vd)

    ##

    def props(self, user):
        wp = None
        if self.widget:
            wp = gws.props_of(self.widget, user, self)
            if not user.can_write(self):
                wp.readOnly = True

        return Props(
            attributeType=self.attributeType,
            name=self.name,
            title=self.title,
            type=self.extType,
            widget=wp,
            uid=self.uid,
            relatedModelUids=[
                m.uid
                for m in self.related_models()
                if user.can_read(m)
            ],
        )

    ##

    def do_validate(self, feature, mc):
        for vd in self.validators:
            if mc.op in vd.ops and not vd.validate(self, feature, mc):
                feature.errors.append(gws.ModelValidationError(
                    fieldName=self.name,
                    message=vd.message,
                ))
                break

    def related_models(self):
        return []

    def find_relatable_features(self, search, mc):
        return []

    def raw_to_python(self, feature, value, mc: gws.ModelContext):
        return value

    def prop_to_python(self, feature, value, mc: gws.ModelContext):
        return value

    def python_to_raw(self, feature, value, mc: gws.ModelContext):
        return value

    def python_to_prop(self, feature, value, mc: gws.ModelContext):
        return value

    ##

    def describe(self):
        desc = self.model.describe()
        if desc:
            return desc.columnMap.get(self.name)
