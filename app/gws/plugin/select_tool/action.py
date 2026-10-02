"""Select action."""

from typing import Optional

import gws
import gws.base.action
import gws.base.web
import gws.base.storage
import gws.lib.uom


@gws.ext.config.action('select')
class Config(gws.base.action.Config):
    """Action for selecting features on the map."""

    storage: Optional[gws.base.storage.Config]
    """Storage for saving and loading selections."""
    tolerance: Optional[gws.UomValueStr]
    """Click tolerance for feature selection."""


@gws.ext.props.action('select')
class Props(gws.base.action.Props):
    storage: gws.base.storage.Props
    tolerance: str


@gws.ext.object.action('select')
class Object(gws.base.action.Object):
    storage: Optional[gws.base.storage.Object]
    tolerance: Optional[gws.UomValue]

    def configure(self):
        self.storage = self.create_child_if_configured(
            gws.base.storage.Object, self.cfg('storage'), categoryName='Select')
        self.tolerance = self.cfg('tolerance')

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            storage=self.storage,
            tolerance=gws.lib.uom.to_str(self.tolerance) if self.tolerance else None,
        )

    @gws.ext.command.api('selectStorage')
    def handle_storage(self, req: gws.WebRequester, p: gws.base.storage.Request) -> gws.base.storage.Response:
        if not self.storage:
            raise gws.NotFoundError('no storage configured')
        return self.storage.handle_request(req, p)
