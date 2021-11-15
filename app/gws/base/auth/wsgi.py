import gws
import gws.base.web.error
import gws.base.auth.error
import gws.base.web.wsgi
import gws.types as t

from . import error
from gws.base.web.wsgi import WebResponse


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

    def require(self, klass, uid):
        try:
            return self.user.require(klass, uid)
        except gws.base.auth.error.ObjectNotFound:
            gws.log.error('require: not found', klass, uid)
            raise gws.base.web.error.NotFound()
        except gws.base.auth.error.AccessDenied:
            gws.log.error('require: denied', klass, uid)
            raise gws.base.web.error.Forbidden()

    def require_project(self, uid):
        return t.cast(gws.IProject, self.require('gws.base.project', uid))

    def require_layer(self, uid):
        return t.cast(gws.ILayer, self.require('gws.ext.layer', uid))

    def acquire(self, klass, uid):
        return self.user.acquire(klass, uid)
