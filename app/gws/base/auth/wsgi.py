import gws
import gws.types as t
import gws.base.web.wsgi
from gws.base.web.wsgi import WebResponse
import gws.base.web.error

from . import core, error


class WebRequest(gws.base.web.wsgi.WebRequest, gws.IWebRequest):
    session: core.Session = t.cast(core.Session, None)

    @property
    def auth(self) -> core.Manager:
        return t.cast(core.Manager, getattr(self.root.application, 'auth'))

    @property
    def user(self) -> core.User:
        if not self.session:
            raise gws.Error('session not opened')
        return self.session.user

    def auth_open(self):
        try:
            self.session = self.auth.open_session(self)
        except error.Error as e:
            raise gws.base.web.error.Forbidden() from e

    def auth_close(self, res: WebResponse):
        self.session = self.auth.close_session(self.session, self, res)

    def require(self, klass: str, uid: t.Optional[str]) -> gws.IObject:
        node = self.root.find(klass, uid)
        if not node:
            gws.log.error('require: not found', klass, uid)
            raise gws.base.web.error.NotFound()
        if not self.user.can_use(node):
            gws.log.error('require: denied', klass, uid)
            raise gws.base.web.error.Forbidden()
        return node

    def require_project(self, uid: t.Optional[str]) -> gws.IProject:
        return t.cast(gws.IProject, self.require('gws.base.project', uid))

    def require_layer(self, uid: t.Optional[str]) -> gws.ILayer:
        return t.cast(gws.ILayer, self.require('gws.ext.layer', uid))

    def acquire(self, klass: str, uid: t.Optional[str]) -> t.Optional[gws.IObject]:
        obj = self.root.find(klass, uid)
        if obj and self.user.can_use(obj):
            return obj
