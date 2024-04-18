"""Provide configuration for the client Dimension module."""

from typing import Optional

import gws
import gws.base.action
import gws.base.storage

gws.ext.new.action('dimension')


class Config(gws.base.action.Config):
    """Dimension action"""

    layerUids: Optional[list[str]]
    """snap layer uids"""
    pixelTolerance: int = 10
    """pixel tolerance"""
    storage: Optional[gws.base.storage.Config]
    """storage configuration"""


class Props(gws.base.action.Props):
    layerUids: Optional[list[str]]
    pixelTolerance: int
    storage: gws.base.storage.Props


class Object(gws.base.action.Object):
    storage: Optional[gws.base.storage.Object]

    def configure(self):
        self.storage = self.create_child_if_configured(
            gws.base.storage.Object, self.cfg('storage'), categoryName='Dimension')

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            layerUids=self.cfg('layerUids') or [],
            pixelTolerance=self.cfg('pixelTolerance'),
            storage=self.storage,
        )

    @gws.ext.command.api('dimensionStorage')
    def handle_storage(self, req: gws.WebRequester, p: gws.base.storage.Request) -> gws.base.storage.Response:
        if not self.storage:
            raise gws.NotFoundError()
        return self.storage.handle_request(req, p)
