"""The ``auth`` action, the client API of the web authentication method."""

from typing import cast

import gws
import gws.base.action

from . import core


@gws.ext.config.action('auth')
class Config(gws.base.action.Config):
    """Login, logout and session checks for web-based authentication."""

    pass


@gws.ext.props.action('auth')
class Props(gws.base.action.Props):
    pass


@gws.ext.object.action('auth')
class Object(gws.base.action.Object):
    """Action for web logins, logouts and session checks.

    Provides the client API of the web authentication method and passes the
    requests to that method.
    """

    method: core.Object
    """The web authentication method."""

    def configure(self):
        m = self.configure_method()
        if not m:
            raise gws.Error('web authorization method required')
        self.method = m

    def configure_method(self):
        """Find the web authentication method.

        Returns:
            The first configured method of type ``web``, or ``None``.
        """
        for m in self.root.app.authMgr.methods:
            if m.extType == 'web':
                return cast(core.Object, m)

    @gws.ext.command.api('authCheck')
    def check(self, req: gws.WebRequester, p: gws.Request) -> core.UserResponse:
        """Return the current user."""
        if req.user.isGuest:
            return core.UserResponse(user=None)
        return core.UserResponse(user=gws.props_of(req.user, req.user))

    @gws.ext.command.api('authLogin')
    def login(self, req: gws.WebRequester, p: core.LoginRequest) -> core.LoginResponse:
        """Log in with a username and a password."""
        return self.method.handle_login(req, p)

    @gws.ext.command.api('authMfaVerify')
    def mfa_verify(self, req: gws.WebRequester, p: core.MfaVerifyRequest) -> core.LoginResponse:
        """Verify a multi-factor code and complete the login."""
        return self.method.handle_mfa_verify(req, p)

    @gws.ext.command.api('authMfaRestart')
    def mfa_restart(self, req: gws.WebRequester, p: gws.Request) -> core.LoginResponse:
        """Restart the multi-factor transaction, for example to send a new code."""
        return self.method.handle_mfa_restart(req, p)

    @gws.ext.command.api('authLogout')
    def logout(self, req: gws.WebRequester, p: gws.Request) -> core.LogoutResponse:
        """Log out the current user."""
        return self.method.handle_logout(req)
