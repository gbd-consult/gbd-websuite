"""Annotate action."""

import gws
import gws.base.action
import gws.base.web
import gws.base.storage
import gws.types as t

gws.ext.new.action('annotate')


class Config(gws.base.action.Config):
    storageUid: t.Optional[str]
    """storage provider uid"""
    storageCategory: t.Optional[gws.base.storage.CategoryConfig]


class Props(gws.base.action.Props):
    storageState: t.Optional[gws.base.storage.State]


class Object(gws.base.action.Object):
    storageProvider: t.Optional[gws.base.storage.provider.Object]
    storageCategoryName: str

    def configure(self):
        self.storageProvider = gws.base.storage.provider.get_for(self)
        if self.storageProvider:
            p = self.cfg('storageCategory')
            self.storageCategoryName = self.storageProvider.add_category(p) if p else 'Annotate'

    def props(self, user):
        p = super().props(user)
        if self.storageProvider:
            p.storageState = self.storageProvider.get_state(self.storageCategoryName, user)
        return p

    @gws.ext.command.api('annotateStorage')
    def storage(self, req: gws.IWebRequester, p: gws.base.storage.Request) -> gws.base.storage.Response:
        if not self.storageProvider:
            raise gws.base.web.error.NotFound()
        return self.storageProvider.handle_request(self.storageCategoryName, req, p)
