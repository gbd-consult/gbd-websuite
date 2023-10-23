"""Validator for required values."""

import gws
import gws.base.model.validator
import gws.types as t

gws.ext.new.modelValidator('required')


class Config(gws.base.model.validator.Config):
    pass


class Object(gws.base.model.validator.Object):
    def validate(self, field, feature, mc):
        val = feature.attributes.get(field.name)

        if isinstance(val, str) and len(val.strip()) > 0:
            return True
        if val is not None:
            return True
        if mc.op == gws.ModelOperation.create and field.isAuto:
            return True

        return False
