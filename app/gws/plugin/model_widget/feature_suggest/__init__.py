"""Feature suggest widget."""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('featureSuggest')
class Config(gws.base.model.widget.Config):
    """Feature suggest widget configuration."""

    pass


@gws.ext.props.modelWidget('featureSuggest')
class Props(gws.base.model.widget.Props):
    pass


@gws.ext.object.modelWidget('featureSuggest')
class Object(gws.base.model.widget.Object):
    pass
