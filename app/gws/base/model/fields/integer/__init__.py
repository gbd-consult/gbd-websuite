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
