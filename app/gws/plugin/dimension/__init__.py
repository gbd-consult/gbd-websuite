"""Provide configuration for the client Dimension module."""

from typing import Optional

import gws
import gws.base.action
import gws.base.storage


@gws.ext.config.action('dimension')
class Config(gws.base.action.Config):
    """Tool for drawing dimensions on the map."""

    layerUids: Optional[list[str]]
    """UIDs of layers to snap to."""
    pixelTolerance: int = 10
    """Snapping distance in screen pixels."""
    storage: Optional[gws.base.storage.Config]
    """Storage for saved dimensions."""


@gws.ext.props.action('dimension')
class Props(gws.base.action.Props):
    layerUids: Optional[list[str]]
    pixelTolerance: int
    storage: gws.base.storage.Props


@gws.ext.object.action('dimension')
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
