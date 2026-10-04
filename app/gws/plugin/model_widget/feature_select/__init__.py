"""Feature select widget.

Drop-down list for choosing the related feature of a ``relatedFeature``
field, optionally with a search field. This is the default widget of
``relatedFeature`` fields.

Example::

    fields+ {
        name "category"
        type "relatedFeature"
        fromColumn "category_id"
        toModel "model_category"
        widget { type "featureSelect" withSearch true }
    }
"""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('featureSelect')
class Config(gws.base.model.widget.Config):
    """Drop-down list for choosing a related feature."""

    withSearch: bool = False
    """Show a search field to filter the list."""


@gws.ext.props.modelWidget('featureSelect')
class Props(gws.base.model.widget.Props):
    withSearch: bool


@gws.ext.object.modelWidget('featureSelect')
class Object(gws.base.model.widget.Object):
    """Feature select widget object."""

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            withSearch=self.cfg('withSearch', default=False),
        )
