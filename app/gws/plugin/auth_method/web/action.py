"""Check user logins and logouts."""

import gws
import gws.base.action
import gws.base.auth.user
import gws.types as t

from . import core

gws.ext.new.action('auth')


class Config(gws.Config):
    pass


class Props(gws.Props):
    pass


class Response(gws.Response):
    user: gws.base.auth.user.Props


class LoginRequest(gws.Request):
    username: str
    password: str


class Object(gws.base.action.Object):
    method: t.Optional[core.Object]

    def configure(self):
        self.method = self.configure_method()
        if not self.method:
            raise gws.Error('web authorization method required')

    def configure_method(self):
        for m in self.root.app.authMgr.methods:
            if m.extType == 'web':
                return t.cast(core.Object, m)

    @gws.ext.command.api('authCheck')
    def check(self, req: gws.IWebRequester, p: gws.Request) -> Response:
        return self._response(req)

    @gws.ext.command.api('authLogin')
    def login(self, req: gws.IWebRequester, p: LoginRequest) -> Response:
        self.method.handle_login(req, p)
        return self._response(req)

    @gws.ext.command.api('authLogout')
    def logout(self, req: gws.IWebRequester, p: gws.Request) -> Response:
        self.method.handle_logout(req)
        return self._response(req)

    def _response(self, req):
        user = req.user
        if user.isGuest:
            return Response(user=None)
        res = Response(user=gws.props(user, user))
        return res
