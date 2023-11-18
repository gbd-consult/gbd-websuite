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
    labels: t.Optional[dict]
    """default label templates"""


class Props(gws.base.action.Props):
    storage: gws.base.storage.Props
    labels: dict


class Object(gws.base.action.Object):
    storage: t.Optional[gws.base.storage.Object]
    labels: t.Optional[dict]

    def configure(self):
        self.storage = self.create_child_if_configured(
            gws.base.storage.Object, self.cfg('storage'), categoryName='Annotate')
        self.labels = self.cfg('labels')

    def props(self, user):
        return gws.merge(
            super().props(user),
            storage=self.storage,
            labels=self.labels,
        )

    @gws.ext.command.api('annotateStorage')
    def handle_storage(self, req: gws.IWebRequester, p: gws.base.storage.Request) -> gws.base.storage.Response:
        if not self.storage:
            raise gws.base.web.error.NotFound()
        return self.storage.handle_request(req, p)
