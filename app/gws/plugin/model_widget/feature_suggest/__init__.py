"""Feature suggest widget."""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('featureSuggest')
class Config(gws.base.model.widget.Config):
    """Input that suggests related features as the user types."""

    pass


@gws.ext.props.modelWidget('featureSuggest')
class Props(gws.base.model.widget.Props):
    pass


@gws.ext.object.modelWidget('featureSuggest')
class Object(gws.base.model.widget.Object):
    pass
