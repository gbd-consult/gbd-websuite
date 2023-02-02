"""Integer field."""

import gws
import gws.base.database.model
import gws.types as t

from .. import scalar


@gws.ext.config.modelField('integer')
class Config(scalar.Config):
    pass


@gws.ext.props.modelField('integer')
class Props(scalar.Props):
    pass


@gws.ext.object.modelField('integer')
class Object(scalar.Object):
    """Integer field object."""
    attributeType = gws.AttributeType.int

    def load_from_props(self, feature, props, user, **kwargs):
        if self.name not in props.attributes:
            return
        val = props.attributes.get(self.name)
        feature.attributes[self.name] = _int(val)

    def load_from_data(self, feature, data, user, **kwargs):
        if self.name not in data.attributes:
            return
        val = data.attributes.get(self.name)
        feature.attributes[self.name] = _int(val)


def _int(val):
    try:
        return int(val)
    except ValueError:
        return gws.ErrorValue
