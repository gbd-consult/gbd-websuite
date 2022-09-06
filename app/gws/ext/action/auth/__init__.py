"""Check user logins and logouts."""

import gws.types as t

import gws.common.action


class Config(t.WithTypeAndAccess):
    """Authorization action"""
    pass


class Response(t.Response):
    user: t.UserProps
    mfaOptions: t.Optional[dict]


class LoginParams(t.Params):
    username: str
    password: str


class MfaVerifyParams(t.Params):
    token: str


class Object(gws.common.action.Object):

    def api_check(self, req: t.IRequest, p: t.Params) -> Response:
        """Check the authorization status"""

        web = self.root.application.auth.get_method('web')
        return web.action_response(req)

    def api_login(self, req: t.IRequest, p: LoginParams) -> Response:
        """Perform a login"""

        web = self.root.application.auth.get_method('web')
        return web.action_login(req, p.username, p.password)

    def api_logout(self, req: t.IRequest, p: t.Params) -> Response:
        """Perform a logout"""

        web = self.root.application.auth.get_method('web')
        return web.action_logout(req)

    def api_mfa_verify(self, req: t.IRequest, p: MfaVerifyParams) -> Response:
        """Verify an MFA token"""

        web = self.root.application.auth.get_method('web')
        return web.action_mfa_verify(req, p)

    def api_mfa_restart(self, req: t.IRequest, p: t.Params) -> Response:
        """Regenerate an MFA token"""

        web = self.root.application.auth.get_method('web')
        return web.action_mfa_restart(req)
