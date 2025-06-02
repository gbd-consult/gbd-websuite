"""Check user logins and logouts."""

from typing import cast

import gws
import gws.base.action

from . import core

gws.ext.new.action('auth')


class Config(gws.base.action.Config):
    """Web-based authorization action configuration."""

    pass


class Props(gws.base.action.Props):
    pass


class Object(gws.base.action.Object):
    method: core.Object

    def configure(self):
        m = self.configure_method()
        if not m:
            raise gws.Error('web authorization method required')
        self.method = m

    def configure_method(self):
        for m in self.root.app.authMgr.methods:
            if m.extType == 'web':
                return cast(core.Object, m)

    @gws.ext.command.api('authCheck')
    def check(self, req: gws.WebRequester, p: gws.Request) -> core.UserResponse:
        if req.user.isGuest:
            return core.UserResponse(user=None)
        return core.UserResponse(user=gws.props_of(req.user, req.user))

    @gws.ext.command.api('authLogin')
    def login(self, req: gws.WebRequester, p: core.LoginRequest) -> core.LoginResponse:
        return self.method.handle_login(req, p)

    @gws.ext.command.api('authMfaVerify')
    def mfa_verify(self, req: gws.WebRequester, p: core.MfaVerifyRequest) -> core.LoginResponse:
        return self.method.handle_mfa_verify(req, p)

    @gws.ext.command.api('authMfaRestart')
    def mfa_restart(self, req: gws.WebRequester, p: gws.Request) -> core.LoginResponse:
        return self.method.handle_mfa_restart(req, p)

    @gws.ext.command.api('authLogout')
    def logout(self, req: gws.WebRequester, p: gws.Request) -> core.LogoutResponse:
        return self.method.handle_logout(req)
