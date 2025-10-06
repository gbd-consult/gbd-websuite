"""Generic scalar field."""

from typing import Optional, Callable, cast

import gws
import gws.base.model.field


class Config(gws.base.model.field.Config):
    """Configuration for the scalar field."""

    isVirtual: Optional[bool]
    """The field is not stored in the DB."""


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Object):
    isVirtual: bool

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
        for mv in self.values:
            if mc.op in mv.ops and mc.user.can_use(mv):
                return mv
