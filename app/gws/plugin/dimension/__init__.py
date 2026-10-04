"""Dimension tool.

Client tool for drawing dimension lines on the map. While drawing, the pointer
snaps to the vertices of the features of the layers given by ``layerUids``,
within ``pixelTolerance`` screen pixels. Dimensions are listed in a sidebar,
where they can be labeled and deleted, and are printed with the map. If a
storage is configured, they can be saved on the server and loaded again.

This module contains the ``dimension`` action. It passes the snapping settings
and the storage settings to the client and implements the API command
``dimensionStorage``, which reads and writes saved dimensions. The client part
(``Sidebar.Dimension``, ``Toolbar.Dimension``) is in ``js``.

Example::

    actions+ {
        type "dimension"
        layerUids ["my_wfs_layer"]
        pixelTolerance 10
        storage {
            permissions {
                read "allow all"
                write "allow all"
                create "allow all"
            }
        }
    }

    client.addElements+ { tag "Sidebar.Dimension" }
    client.addElements+ { tag "Toolbar.Dimension" }
"""

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
    """Dimension action."""

    storage: Optional[gws.base.storage.Object]
    """Storage for saved dimensions, or ``None`` if not configured."""

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
        """Read or write saved dimensions."""
        if not self.storage:
            raise gws.NotFoundError()
        return self.storage.handle_request(req, p)
