"""Annotate action."""

import gws
import gws.base.action
import gws.base.web
import gws.base.storage
import gws.types as t

gws.ext.new.action('annotate')


class Config(gws.base.action.Config):
    storage: t.Optional[gws.base.storage.Config]
    """storage configuration"""


class Props(gws.base.action.Props):
    storage: gws.base.storage.Props


class Object(gws.base.action.Object):
    storage: t.Optional[gws.base.storage.Object]

    def configure(self):
        self.storage = self.create_child_if_configured(
            gws.base.storage.Object, self.cfg('storage'), categoryName='Annotate')

    def props(self, user):
        return gws.merge(
            super().props(user),
            storage=self.storage,
        )

    @gws.ext.command.api('annotateStorage')
    def handle_storage(self, req: gws.IWebRequester, p: gws.base.storage.Request) -> gws.base.storage.Response:
        if not self.storage:
            raise gws.base.web.error.NotFound()
        return self.storage.handle_request(req, p)
