"""Hidden input widget.

The field is not shown in the feature form.

Example::

    fields+ { name "internal_id" type "text" widget.type "hidden" }
"""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('hidden')
class Config(gws.base.model.widget.Config):
    """Field that is not shown in the form."""

    pass


@gws.ext.props.modelWidget('hidden')
class Props(gws.base.model.widget.Props):
    pass


@gws.ext.object.modelWidget('hidden')
class Object(gws.base.model.widget.Object):
    """Hidden input widget object."""

    pass
