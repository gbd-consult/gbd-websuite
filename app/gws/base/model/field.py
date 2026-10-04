"""Base model field."""

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
    """Field name, matching the column or attribute name in the source."""
    title: Optional[str]
    """Field title shown in the client."""

    isPrimaryKey: Optional[bool]
    """The field is a primary key."""
    isRequired: Optional[bool]
    """The field must not be empty."""
    isUnique: Optional[bool]
    """The field is unique."""
    isAuto: Optional[bool]
    """The value is set by the database and never written."""
    isHidden: Optional[bool]
    """The field is not automatically displayed in the UI."""

    values: Optional[list[gws.ext.config.modelValue]]
    """Value sources that compute the field value on read, create or update."""
    validators: Optional[list[gws.ext.config.modelValidator]]
    """Additional validators for the field value."""

    widget: Optional[gws.ext.config.modelWidget]
    """Widget that displays and edits the field in the client."""


##


class Object(gws.ModelField):
    """Base model field.

    Provides the field flags, value objects, validators, widget, props and validation.
    Subclasses implement reading and writing the field value and the conversion between
    record, feature and props values.
    """

    notEmptyValidator: gws.ModelValidator
    """Validator that checks that the value is not empty."""
    formatValidator: gws.ModelValidator
    """Validator that checks the format of the value."""

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
        """Set the field flags from the configuration or the source column.

        ``isPrimaryKey``, ``isRequired`` and ``isAuto`` default to the properties of the
        source column with the field name. ``isUnique`` and ``isHidden`` default to False.
        """
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

        self.isHidden = self.cfg('isHidden', default=False)

    def configure_values(self):
        """Create the configured value objects.

        Returns:
            True if values are configured, None otherwise.
        """
        p = self.cfg('values')
        if p:
            self.values = self.create_children(gws.ext.object.modelValue, p)
            return True

    def configure_validators(self):
        """Create the configured validators.

        Configured ``notEmpty`` and ``format`` validators replace the shared default ones,
        which apply to ``create`` and ``update``.

        Returns:
            Always True.
        """
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
        """Create a validator as a child of this field.

        Args:
            cfg: The validator configuration.

        Returns:
            The validator object.
        """
        return self.create_child(gws.ext.object.modelValidator, cfg)

    def configure_widget(self):
        """Create the configured widget.

        Returns:
            True if a widget is configured, None otherwise.
        """
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
