"""Feature suggest widget.

Input for choosing the related feature of a ``relatedFeature`` field, which
suggests matching features as the user types.

Example::

    fields+ {
        name "street"
        type "relatedFeature"
        fromColumn "street_id"
        toModel "model_street"
        widget.type "featureSuggest"
    }
"""

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
    """Feature suggest widget object."""

    pass
