"""File list widget.

A feature list widget for related features that hold files. ``toFileField``
names the ``file`` field of the related model that holds the file. The
button options are the same as for the ``featureList`` widget.

Example::

    fields+ {
        name "docs"
        type "relatedFeatureList"
        toModel "model_document"
        toColumn "poi_id"
        widget {
            type "fileList"
            toFileField "documentFile"
            withNewButton true
            withUnlinkButton true
        }
    }
"""

import gws
import gws.base.model.widget
import gws.plugin.model_widget.feature_list as feature_list


@gws.ext.config.modelWidget('fileList')
class Config(feature_list.Config):
    """List of related files."""

    toFileField: str
    """File field of the related model that holds the file."""


@gws.ext.props.modelWidget('fileList')
class Props(gws.base.model.widget.Props):
    withNewButton: bool
    withLinkButton: bool
    withEditButton: bool
    withUnlinkButton: bool
    withDeleteButton: bool
    toFileField: str


@gws.ext.object.modelWidget('fileList')
class Object(feature_list.Object):
    """File list widget object."""

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            toFileField=self.cfg('toFileField'),
        )
