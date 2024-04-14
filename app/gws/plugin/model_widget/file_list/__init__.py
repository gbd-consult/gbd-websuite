"""File list widget."""

import gws
import gws.base.model.widget
import gws.plugin.model_widget.feature_list as feature_list

gws.ext.new.modelWidget('fileList')


class Config(feature_list.Config):
    toFileField: str


class Props(gws.base.model.widget.Props):
    withNewButton: bool
    withLinkButton: bool
    withEditButton: bool
    withUnlinkButton: bool
    withDeleteButton: bool
    toFileField: str


class Object(feature_list.Object):
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            toFileField=self.cfg('toFileField'),
        )
