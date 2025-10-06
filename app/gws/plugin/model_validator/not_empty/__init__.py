"""Validator for non-empty values."""

import gws
import gws.base.model.validator

gws.ext.new.modelValidator('notEmpty')


class Config(gws.base.model.validator.Config):
    """Validator for non-empty values."""

    pass


class Object(gws.base.model.validator.Object):
    def validate(self, field, feature, mc):
        val = feature.attributes.get(field.name)

        if mc.op == gws.ModelOperation.create and field.isAuto:
            return True
        if isinstance(val, str):
            return len(val.strip()) > 0
        if val is not None:
            return True

        return False
