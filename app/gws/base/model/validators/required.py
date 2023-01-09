"""Static value."""

import gws
import gws.base.model.validator
import gws.types as t


@gws.ext.config.modelValidator('required')
class Config(gws.base.model.validator.Config):
    pass


@gws.ext.object.modelValidator('required')
class Object(gws.base.model.validator.Object):
    def validate(self, feature, field, user, **kwargs):
        val = feature.attributes.get(field.name)
        if val is None:
            return False
        if isinstance(val, str) and not val.strip():
            return False
        return True
