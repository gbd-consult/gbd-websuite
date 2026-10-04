"""Feature list widget.

Shows the related features of a ``relatedFeatureList``,
``relatedMultiFeatureList`` or ``relatedLinkedFeatureList`` field as a list.
Buttons to create a new related feature, link an existing one, edit, unlink
or delete the selected one can be turned on and off. This is the default
widget of feature list fields.

Example::

    fields+ {
        name "pois"
        type "relatedFeatureList"
        toModel "model_poi"
        toColumn "category_id"
        widget { type "featureList" withDeleteButton true }
    }
"""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('featureList')
class Config(gws.base.model.widget.Config):
    """List of related features."""

    withNewButton: bool = True
    """Show a button to create a new related feature."""
    withLinkButton: bool = True
    """Show a button to link an existing feature."""
    withEditButton: bool = True
    """Show a button to edit the selected feature."""
    withUnlinkButton: bool = False
    """Show a button to unlink the selected feature without deleting it."""
    withDeleteButton: bool = False
    """Show a button to delete the selected feature."""


@gws.ext.props.modelWidget('featureList')
class Props(gws.base.model.widget.Props):
    withNewButton: bool
    withLinkButton: bool
    withEditButton: bool
    withUnlinkButton: bool
    withDeleteButton: bool


@gws.ext.object.modelWidget('featureList')
class Object(gws.base.model.widget.Object):
    """Feature list widget object."""

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            withNewButton=self.cfg('withNewButton', default=True),
            withLinkButton=self.cfg('withLinkButton', default=True),
            withEditButton=self.cfg('withEditButton', default=True),
            withUnlinkButton=self.cfg('withUnlinkButton', default=False),
            withDeleteButton=self.cfg('withDeleteButton', default=False),
        )
