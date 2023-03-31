"""Provide configuration for the client Dimension module."""

import gws
import gws.base.action
import gws.base.storage
import gws.types as t

gws.ext.new.action('dimension')

STORAGE_CATEGORY = 'Dimension'


class Config(gws.ConfigWithAccess):
    """Dimension action"""

    layers: t.Optional[list[str]]
    """target layer uids"""
    pixelTolerance: int = 10
    """pixel tolerance"""


class Props(gws.base.action.Props):
    layerUids: t.Optional[list[str]]
    pixelTolerance: int


class Object(gws.base.action.Object):

    def props(self, user):
        return gws.merge(
            super().props(user),
            layerUids=self.cfg('layers') or [],
            pixelTolerance=self.cfg('pixelTolerance'),
        )

    @gws.ext.command.api('dimensionStorage')
    def storage(self, req: gws.IWebRequester, p: gws.base.storage.Params) -> gws.base.storage.Response:
        helper: gws.base.storage.Object = self.root.app.require_helper('storage')
        return helper.handle_action(req, p, STORAGE_CATEGORY)
