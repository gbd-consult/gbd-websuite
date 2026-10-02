"""File list widget."""

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
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            toFileField=self.cfg('toFileField'),
        )
