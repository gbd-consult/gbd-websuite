"""Feature select widget."""

import gws
import gws.base.model.widget

gws.ext.new.modelWidget('featureList')


class Config(gws.base.model.widget.Config):
    withNewButton: bool = True
    withLinkButton: bool = True
    withEditButton: bool = True
    withUnlinkButton: bool = False
    withDeleteButton: bool = False


class Props(gws.base.model.widget.Props):
    withNewButton: bool
    withLinkButton: bool
    withEditButton: bool
    withUnlinkButton: bool
    withDeleteButton: bool


class Object(gws.base.model.widget.Object):
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            withNewButton=self.cfg('withNewButton', default=True),
            withLinkButton=self.cfg('withLinkButton', default=True),
            withEditButton=self.cfg('withEditButton', default=True),
            withUnlinkButton=self.cfg('withUnlinkButton', default=False),
            withDeleteButton=self.cfg('withDeleteButton', default=False),
        )
