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
    """Configuration for the model field."""

    name: str
    """The name of the field."""
    title: Optional[str]
    """The title of the field."""

    isPrimaryKey: Optional[bool]
    """If True, the field is a primary key."""
    isRequired: Optional[bool]
    """If True, the field is required."""
    isUnique: Optional[bool]
    """If True, the field is unique."""
    isAuto: Optional[bool]
    """If True, the field is auto-updated."""

    values: Optional[list[gws.ext.config.modelValue]]
    """List of possible values for the field."""
    validators: Optional[list[gws.ext.config.modelValidator]]
    """List of validators for the field."""

    widget: Optional[gws.ext.config.modelWidget]
    """Configuration for the field widget."""


##


class Object(gws.ModelField):
    notEmptyValidator: gws.ModelValidator
    formatValidator: gws.ModelValidator

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
        vd_not_empty = None
        vd_format = None

        for p in self.cfg('validators', default=[]):
            vd = self.create_validator(p)
            if vd.extType == 'notEmpty':
                vd_not_empty = vd
            elif vd.extType == 'format':
                vd_format = vd
            else:
                self.validators.append(vd)

        self.notEmptyValidator = vd_not_empty or self.root.create_shared(
            gws.ext.object.modelValidator,
            type='notEmpty',
            uid='gws.base.model.field.default_validator_notEmpty',
            forCreate=True,
            forUpdate=True,
        )

        self.formatValidator = vd_format or self.root.create_shared(
            gws.ext.object.modelValidator,
            type='format',
            uid='gws.base.model.field.default_validator_format',
            forCreate=True,
            forUpdate=True,
        )

        return True

    def create_validator(self, cfg):
        return self.create_child(gws.ext.object.modelValidator, cfg)

    def configure_widget(self):
        p = self.cfg('widget')
        if p:
            self.widget = self.create_child(gws.ext.object.modelWidget, p)
            return True

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
            relatedModelUids=[m.uid for m in self.related_models() if user.can_read(m)],
        )

    ##

    def do_validate(self, feature, mc):
        # apply the 'notEmpty' validator and exit immediately if it fails
        # (no error message if field is not required)

        ok = self.notEmptyValidator.validate(self, feature, mc)
        if not ok:
            if self.isRequired:
                feature.errors.append(
                    gws.ModelValidationError(
                        fieldName=self.name,
                        message=self.notEmptyValidator.message,
                    )
                )
            return

        # apply the 'format' validator

        ok = self.formatValidator.validate(self, feature, mc)
        if not ok:
            feature.errors.append(
                gws.ModelValidationError(
                    fieldName=self.name,
                    message=self.formatValidator.message,
                )
            )
            return

        # apply others

        for vd in self.validators:
            if mc.op in vd.ops:
                ok = vd.validate(self, feature, mc)
                if not ok:
                    feature.errors.append(
                        gws.ModelValidationError(
                            fieldName=self.name,
                            message=vd.message,
                        )
                    )
                    return

    def related_models(self):
        return []

    def find_relatable_features(self, search, mc):
        return []

    def raw_to_python(self, feature, value, mc):
        return value

    def prop_to_python(self, feature, value, mc):
        return value

    def python_to_raw(self, feature, value, mc):
        return value

    def python_to_prop(self, feature, value, mc):
        return value

    ##

    def describe(self):
        desc = self.model.describe()
        if desc:
            return desc.columnMap.get(self.name)
