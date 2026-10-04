"""Base class for scalar fields."""

from typing import Optional, Callable, cast

import gws
import gws.base.model.field


class Config(gws.base.model.field.Config):
    """Configuration for the scalar field."""

    isVirtual: Optional[bool]
    """The field is not read from or written to the database."""


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Object):
    """Base scalar field.

    Maps the field to the source column with the same name. Provides reading the column
    and writing it to the record, and the transfer of the value between record, feature
    and props, applying value objects and field permissions.
    """

    isVirtual: bool
    """The field is not read from or written to the database."""

    def configure(self):
        self.isVirtual = self.cfg('isVirtual', default=False)

    def before_select(self, mc):
        if self.isVirtual:
            return
        model = cast(gws.DatabaseModel, self.model)
        mc.dbSelect.columns.append(model.column(self.name))

    def after_select(self, features, mc):
        for feature in features:
            self.from_record(feature, mc)

    def before_create(self, feature, mc):
        if self.isAuto:
            return
        if self.isVirtual:
            return
        self.to_record(feature, mc)

    def before_update(self, feature, mc):
        if self.isAuto:
            return
        if self.isVirtual:
            return
        self.to_record(feature, mc)

    ##

    def from_props(self, feature, mc):
        value = feature.props.attributes.get(self.name)
        if value is not None:
            value = self.prop_to_python(feature, value, mc)
        if value is not None:
            feature.set(self.name, value)

    def to_props(self, feature, mc):
        if not mc.user.can_read(self):
            return
        value = feature.get(self.name)
        if value is not None:
            value = self.python_to_prop(feature, value, mc)
        if value is not None:
            feature.props.attributes[self.name] = value

    ##

    def do_init(self, feature, mc):
        value = self.get_value(
            feature,
            {},
            mc.user.can_read(self),
            self.prop_to_python,
            mc,
        )
        if value is not None:
            feature.set(self.name, value)

    ##

    def from_record(self, feature, mc):
        value = self.get_value(
            feature,
            feature.record.attributes,
            mc.user.can_read(self),
            self.raw_to_python,
            mc,
        )
        if value is not None:
            feature.set(self.name, value)

    def to_record(self, feature, mc):
        value = self.get_value(
            feature,
            feature.attributes,
            mc.user.can_write(self),
            self.python_to_raw,
            mc,
        )
        if value is not None:
            feature.record.attributes[self.name] = value

    def get_value(
        self,
        feature: gws.Feature,
        source: dict,
        has_access: bool,
        convert_fn: Callable,
        mc: gws.ModelContext,
    ):
        """Compute the field value for the current operation.

        A value object that is not a default value provides the value. Otherwise the value
        is taken from ``source`` and converted, if the user has access and the value is not
        None. Otherwise a default value object provides the value.

        Args:
            feature: The feature.
            source: A dict of attributes to read the value from.
            has_access: Whether the user may access the field.
            convert_fn: Function ``(feature, value, mc)`` that converts the source value.
            mc: The model context.

        Returns:
            The value, or None if there is none.
        """
        mv = self.model_value(mc)

        if mv and not mv.isDefault:
            return mv.compute(self, feature, mc)

        if has_access and self.name in source:
            value = source.get(self.name)
            if value is not None:
                return convert_fn(feature, value, mc)

        if mv:
            return mv.compute(self, feature, mc)

    def model_value(self, mc: gws.ModelContext):
        """Return the first value object for the current operation that the user can use.

        Args:
            mc: The model context.

        Returns:
            The value object, or None if there is none.
        """
        for mv in self.values:
            if mc.op in mv.ops and mc.user.can_use(mv):
                return mv
