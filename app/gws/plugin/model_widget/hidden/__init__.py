"""Hidden input widget."""

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
    pass
