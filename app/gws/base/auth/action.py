"""Check user logins and logouts."""

import gws
import gws.base.action
import gws.base.web.error
import gws.types as t

from . import user
from .methods import web


class Response(gws.Response):
    user: user.Props
    mfaFrom: t.Optional[dict]


class LoginRequest(gws.Request):
    username: str
    password: str


class MfaVerifyRequest(gws.Request):
    otp: str


@gws.ext.object.action('auth')
class Object(gws.base.action.Object):
    webMethod: t.Optional[web.Object]

    def configure(self):
        for m in self.root.app.authMgr.methods:
            if m.extType == 'web':
                self.webMethod = t.cast(web.Object, m)
                break
        if not self.webMethod:
            raise gws.Error('web authorization method required')

    @gws.ext.command.api('authCheck')
    def check(self, req: gws.IWebRequester, p: gws.Request) -> Response:
        return self._response(req)

    @gws.ext.command.api('authLogin')
    def login(self, req: gws.IWebRequester, p: LoginRequest) -> Response:
        self.webMethod.action_login(req, p)
        return self._response(req)

    @gws.ext.command.api('authLogout')
    def logout(self, req: gws.IWebRequester, p: gws.Request) -> Response:
        self.webMethod.action_logout(req)
        return self._response(req)

    @gws.ext.command.api('authMfaStart')
    def mfa_start(self, req: gws.IWebRequester, p: gws.Request) -> Response:
        self.webMethod.action_mfa_start(req)
        return self._response(req)

    @gws.ext.command.api('authMfaVerify')
    def mfa_verify(self, req: gws.IWebRequester, p: MfaVerifyRequest) -> Response:
        self.webMethod.action_mfa_verify(req, p)
        return self._response(req)

    @gws.ext.command.api('authMfaRestart')
    def mfa_restart(self, req: gws.IWebRequester, p: MfaVerifyRequest) -> Response:
        self.webMethod.action_mfa_restart(req, p)
        return self._response(req)

    def _response(self, req):
        usr = req.user
        if usr.isGuest:
            return Response(user=None)
        res = Response(usr=gws.props(usr, usr))
        if usr.pendingMfa:
            res.mfaFrom = usr.pendingMfa.form
            res.mfaError = usr.pendingMfa.error or None
        return res
