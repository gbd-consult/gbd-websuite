"""Select action."""

import gws
import gws.base.action
import gws.base.web
import gws.base.storage
import gws.lib.uom
import gws.types as t

gws.ext.new.action('select')


class Config(gws.base.action.Config):
    storage: t.Optional[gws.base.storage.Config]
    """storage configuration"""
    tolerance: t.Optional[gws.Measurement]
    """click tolerance"""


class Props(gws.base.action.Props):
    storage: gws.base.storage.Props
    tolerance: str


class Object(gws.base.action.Object):
    storage: t.Optional[gws.base.storage.Object]
    tolerance: t.Optional[gws.Measurement]

    def configure(self):
        self.storage = self.create_child_if_configured(
            gws.base.storage.Object, self.cfg('storage'), categoryName='Select')
        self.tolerance = self.cfg('tolerance')

    def props(self, user):
        return gws.merge(
            super().props(user),
            storage=self.storage,
            tolerance=gws.lib.uom.to_str(self.tolerance) if self.tolerance else None,
        )

    @gws.ext.command.api('selectStorage')
    def handle_storage(self, req: gws.IWebRequester, p: gws.base.storage.Request) -> gws.base.storage.Response:
        if not self.storage:
            raise gws.base.web.error.NotFound()
        return self.storage.handle_request(req, p)
