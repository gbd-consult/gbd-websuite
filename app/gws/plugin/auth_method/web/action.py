"""Check user logins and logouts."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.auth.user

from . import core

gws.ext.new.action('auth')


class Config(gws.base.action.Config):
    pass


class Props(gws.base.action.Props):
    pass


class Response(gws.Response):
    user: gws.base.auth.user.Props


class LoginRequest(gws.Request):
    username: str
    password: str


class Object(gws.base.action.Object):
    method: Optional[core.Object]

    def configure(self):
        self.method = self.configure_method()
        if not self.method:
            raise gws.Error('web authorization method required')

    def configure_method(self):
        for m in self.root.app.authMgr.methods:
            if m.extType == 'web':
                return cast(core.Object, m)

    @gws.ext.command.api('authCheck')
    def check(self, req: gws.WebRequester, p: gws.Request) -> Response:
        return self._response(req)

    @gws.ext.command.api('authLogin')
    def login(self, req: gws.WebRequester, p: LoginRequest) -> Response:
        self.method.handle_login(req, p)
        return self._response(req)

    @gws.ext.command.api('authLogout')
    def logout(self, req: gws.WebRequester, p: gws.Request) -> Response:
        self.method.handle_logout(req)
        return self._response(req)

    def _response(self, req):
        user = req.user
        if user.isGuest:
            return Response(user=None)
        res = Response(user=gws.u.make_props(user, user))
        return res
