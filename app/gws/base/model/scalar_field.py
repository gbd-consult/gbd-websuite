"""Generic scalar field."""

import gws
import gws.types as t

from . import field


class Config(field.Config):
    pass


class Props(field.Props):
    pass


class Object(field.Object, gws.IModelScalarField):
    def before_select(self, mc):
        model = t.cast(gws.IDatabaseModel, self.model)
        mc.dbSelect.columns.append(model.column(self.name))

    def after_select(self, features, mc):
        self.load_from_record(features, mc)

    def before_create(self, features, mc):
        self.store_to_record(features, mc)

    def before_update(self, features, mc):
        self.store_to_record(features, mc)

    ##

    def raw_to_python(self, value, mc: gws.ModelContext):
        return value

    def prop_to_python(self, value, mc: gws.ModelContext):
        return value

    def python_to_raw(self, value, mc: gws.ModelContext):
        return value

    def python_to_prop(self, value, mc: gws.ModelContext):
        return value

    ##

    def from_props(self, features, mc):
        for feature in features:
            data = feature.props.attributes.get(self.name)
            if data is not None:
                data = self.prop_to_python(data, mc)
            if data is not None:
                feature.attributes[self.name] = data

    def to_props(self, features, mc):
        if not mc.user.can_read(self):
            return

        for feature in features:
            data = feature.get(self.name)
            if data is not None:
                data = self.python_to_prop(data, mc)
            if data is not None:
                feature.props.attributes[self.name] = data

    ##

    def do_init_with_props(self, feature, mc):
        self.from_props([feature], mc)

    def do_init_with_record(self, feature, mc):
        self.load_from_record([feature], mc)

    ##

    def load_from_record(self, features, mc):
        mv = self.get_value(mc)
        for feature in features:
            data = self.get_data(
                feature,
                feature.record.attributes,
                mc.user.can_read(self),
                mv,
                self.raw_to_python,
                mc
            )
            if data is not None:
                feature.set(self.name, data)

    def store_to_record(self, features, mc):
        mv = self.get_value(mc)
        for feature in features:
            data = self.get_data(
                feature,
                feature.attributes,
                mc.user.can_write(self),
                mv,
                self.python_to_raw,
                mc
            )
            if data is not None:
                feature.record.attributes[self.name] = data

    def get_value(self, mc: gws.ModelContext):
        for mv in self.values:
            if mc.mode in mv.modes and mc.user.can_use(mv):
                return mv

    def get_data(
            self,
            feature: gws.IFeature,
            attributes: dict,
            has_access: bool,
            mv: gws.IModelValue,
            convert_fn: t.Callable,
            mc: gws.ModelContext
    ):
        if mv and not mv.isDefault:
            return mv.compute(self, feature, mc)

        if has_access and self.name in attributes:
            data = attributes.get(self.name)
            if data is not None:
                return convert_fn(data, mc)
            return

        if mv:
            return mv.compute(self, feature, mc)
