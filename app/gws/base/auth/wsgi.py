import gws
import gws.base.web.error
import gws.base.web.wsgi
import gws.types as t
from gws.base.web.wsgi import WebResponse
from . import error


class WebRequest(gws.base.web.wsgi.WebRequest, gws.IWebRequest):
    session: gws.IAuthSession = None  # type: ignore

    @property
    def auth(self):
        return self.root.application.auth

    @property
    def user(self) -> gws.IUser:
        if not self.session:
            raise gws.Error('session not opened')
        return self.session.user

    def auth_open(self):
        try:
            self.session = self.auth.open_session(self)
            gws.log.debug(f'auth_open: typ={self.session.typ!r} user={self.session.user.uid!r}')
        except error.Error as e:
            raise gws.base.web.error.Forbidden() from e

    def auth_close(self, res: WebResponse):
        gws.log.debug(f'auth_close: typ={self.session.typ!r} user={self.session.user.uid!r}')
        self.session = self.auth.close_session(self.session, self, res)

    def require(self, klass: str, uid: t.Optional[str]) -> gws.IObject:
        obj = self.root.find(klass, uid)
        if not obj:
            gws.log.error('require: not found', klass, uid)
            raise gws.base.web.error.NotFound()
        if not self.user.can_use(obj):
            gws.log.error('require: denied', klass, uid)
            raise gws.base.web.error.Forbidden()
        return obj

    def require_project(self, uid: t.Optional[str]) -> gws.IProject:
        return t.cast(gws.IProject, self.require('gws.base.project.core', uid))

    def require_layer(self, uid: t.Optional[str]) -> gws.ILayer:
        return t.cast(gws.ILayer, self.require('gws.ext.layer', uid))

    def acquire(self, klass: str, uid: t.Optional[str]) -> t.Optional[gws.IObject]:
        obj = self.root.find(klass, uid)
        if obj and self.user.can_use(obj):
            return obj
