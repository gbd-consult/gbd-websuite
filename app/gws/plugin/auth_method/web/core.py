"""The ``web`` authentication method and its API types."""

import re
from typing import Optional, cast

import gws
import gws.base.auth
import gws.base.web


class LoginRedirectRule(gws.Data):
    """Redirect to a login page for page requests that are denied."""

    pattern: Optional[gws.Regex]
    """Regular expression for URLs to redirect."""
    target: str
    """Login page URL."""


@gws.ext.config.authMethod('web')
class Config(gws.base.auth.method.Config):
    """Authentication with a login form and a session cookie."""

    cookieName: str = 'auth'
    """Name of the session cookie."""
    cookiePath: str = '/'
    """Path attribute of the session cookie."""
    cookieSameSite: str = 'Lax'
    """SameSite attribute of the session cookie."""
    loginRedirect: Optional[LoginRedirectRule]
    """Redirect denied page requests to a login page."""


##


class UserResponse(gws.Response):
    """Response with the current user."""

    user: Optional[gws.base.auth.user.Props]
    """Properties of the user, ``None`` for guests."""


class LogoutResponse(gws.Response):
    """Response to a logout request."""

    pass


class LoginRequest(gws.Request):
    """Login request."""

    username: str
    """Login name."""
    password: str
    """Password."""
    to: Optional[str]
    """URL path to return to after the login."""


class LoginResponse(gws.Response):
    """Response to a login or multi-factor request."""

    user: Optional[gws.base.auth.user.Props]
    """Properties of the logged-in user, set when the login is completed without a second step."""
    mfaState: Optional[gws.AuthMultiFactorState]
    """State of the multi-factor transaction, if there is one."""
    mfaMessage: str = ''
    """Message for the user in the multi-factor step."""
    mfaCanRestart: bool = False
    """The multi-factor transaction can be restarted."""
    redirectTo: str = ''
    """URL path to go to after a completed login, empty if none."""


class MfaVerifyRequest(gws.Request):
    """Request to verify a multi-factor code."""

    payload: dict
    """Data entered by the user, for example ``{"code": "123456"}``."""
    to: Optional[str]
    """URL path to return to after the login."""


##

_DELETED_SESSION = 'web:deleted'


@gws.ext.object.authMethod('web')
class Object(gws.base.auth.method.Object):
    """Web authentication method.

    Authenticates users with a login form and keeps their sessions in the
    session manager, identified by a session cookie. Handles logins, logouts
    and multi-factor verification for the ``auth`` action.
    """

    cookieName: str
    """Name of the session cookie."""
    cookiePath: str
    """Path attribute of the session cookie."""
    cookieSameSite: str
    """SameSite attribute of the session cookie."""
    loginRedirect: Optional[LoginRedirectRule]
    """Redirect rule for denied page requests, or ``None``."""

    deletedSession: gws.base.auth.session.Object
    """Placeholder session for requests with an invalid session cookie."""

    def configure(self):
        self.uid = 'gws.plugin.auth.method.web'
        self.cookieName = self.cfg('cookieName', default=Config.cookieName)
        self.cookiePath = self.cfg('cookiePath', default=Config.cookiePath)
        self.cookieSameSite = self.cfg('cookieSameSite', default=Config.cookieSameSite)
        self.loginRedirect = self.cfg('loginRedirect')
        self.root.app.middlewareMgr.register(self, self.uid, depends_on=['auth'])

    ##

    def exit_middleware(self, req, res):
        if res.status in (403, 401) and req.isGet:
            self._check_login_redirect(req, res)

    def _check_login_redirect(self, req: gws.WebRequester, res: gws.WebResponder):
        """Redirect a denied page request to the login page, if configured."""
        lr = self.loginRedirect
        if not lr:
            return
        request_uri = req.env('REQUEST_URI', '')
        if not request_uri:
            return
        if lr.pattern and not re.match(lr.pattern, request_uri):
            return
        redir = req.relative_url_for(lr.target, to=request_uri)
        res.set_status(302)
        res.add_header('Location', redir)
        res.set_body(f'Redirecting to {redir}...')
        gws.log.debug(f'auth web: redirect {res.status=} {redir=}')

    def activate(self):
        am = self.root.app.authMgr
        self.deletedSession = gws.base.auth.session.Object(
            uid=_DELETED_SESSION,
            method=self,
            user=am.guestUser,
        )

    def open_session(self, req):
        am = self.root.app.authMgr

        sid = req.cookie(self.cookieName)
        if not sid:
            return

        sess = am.sessionMgr.get(sid)

        if not sess:
            gws.log.debug(f'open_session: {sid=} not found or invalid')
            return self.deletedSession

        return sess

    def close_session(self, req, res):
        am = self.root.app.authMgr

        sess = getattr(req, 'session')
        if not sess:
            return

        if sess.uid == _DELETED_SESSION:
            gws.log.debug('session cookie=deleted')
            res.delete_cookie(
                self.cookieName,
                path=self.cookiePath,
            )
            return

        if res.status < 400:
            gws.log.debug(f'session cookie={sess.uid!r}')
            res.set_cookie(
                self.cookieName,
                sess.uid,
                path=self.cookiePath,
                secure=self.secure,
                samesite=self.cookieSameSite,
                httponly=True,
            )
            am.sessionMgr.touch(sess)

    def handle_login(self, req: gws.WebRequester, p: LoginRequest) -> LoginResponse:
        """Log in with a username and a password.

        The credentials are checked by the authentication providers. If the
        user has an ``mfaUid``, a multi-factor transaction is started and kept
        in a new guest session; ``handle_mfa_verify`` completes the login.
        Otherwise the current session is replaced by a new session for the user.

        Args:
            req: Web requester.
            p: Login request.

        Returns:
            The user and the redirect target, or the multi-factor state.

        Raises:
            ``gws.ForbiddenError``: If the user is already logged in, the method is secure and the request is not, or the multi-factor transaction cannot be started.
            ``gws.AuthenticationError``: If the credentials are not accepted.
        """
        if not req.user.isGuest:
            raise gws.ForbiddenError(f'login: already logged-in {req.user.uid=}')

        if self.secure and not req.isSecure:
            raise gws.ForbiddenError('login: insecure_context, ignored')

        user = self.root.app.authMgr.authenticate(self, p, req)
        if not user:
            raise gws.AuthenticationError('login: user not found')

        if user.mfaUid:
            mfa = self._mfa_start(req, user)
            gws.log.info(f'LOGGED_IN (MFA pending): {user.uid=} {user.roles=}')
            return self._mfa_response(mfa)

        self._finalize_login(req, user)
        return LoginResponse(user=gws.props_of(user, user), redirectTo=self._redirect_target(p.to))

    def handle_mfa_verify(self, req: gws.WebRequester, p: MfaVerifyRequest) -> LoginResponse:
        """Verify a multi-factor payload.

        On success, the session is replaced by a new session for the user. If
        the adapter allows another attempt, the transaction is kept. Otherwise
        the session is deleted.

        Args:
            req: Web requester.
            p: Verification request.

        Returns:
            The multi-factor state, with the redirect target on success.

        Raises:
            ``gws.ForbiddenError``: If the session has no valid multi-factor transaction.
            ``gws.AuthenticationError``: If the verification failed.
        """
        try:
            mfa = self._mfa_verify(req, p.payload)
        except gws.ForbiddenError:
            self._delete_session(req)
            raise

        if mfa.state == gws.AuthMultiFactorState.ok:
            self._finalize_login(req, mfa.user)
            return self._mfa_response(mfa, self._redirect_target(p.to))

        if mfa.state == gws.AuthMultiFactorState.retry:
            return self._mfa_response(mfa)

        self._delete_session(req)
        raise gws.AuthenticationError(f'MFA: verify failed {mfa.state=}')

    def handle_mfa_restart(self, req: gws.WebRequester, p: gws.Request) -> LoginResponse:
        """Restart the multi-factor transaction of the current session.

        Args:
            req: Web requester.
            p: Request parameters.

        Returns:
            The state of the new transaction.

        Raises:
            ``gws.ForbiddenError``: If the session has no valid transaction or it cannot be restarted. The session is deleted in this case.
        """
        try:
            mfa = self._mfa_restart(req)
        except gws.ForbiddenError:
            self._delete_session(req)
            raise

        return self._mfa_response(mfa)

    def handle_logout(self, req: gws.WebRequester) -> LogoutResponse:
        """Log out the current user and delete the session.

        Args:
            req: Web requester.

        Returns:
            An empty response.

        Raises:
            ``gws.ForbiddenError``: If the session was opened by a different method.
        """
        if req.user.isGuest:
            self._delete_session(req)
            return LogoutResponse()

        if req.session.method != self:
            raise gws.ForbiddenError(f'wrong method for logout: {req.session.method!r}')

        self._delete_session(req)

        gws.log.info(f'LOGGED_OUT: user={req.user.uid!r}')
        return LogoutResponse()

    ##

    def _delete_session(self, req: gws.WebRequester):
        """Delete the current session and replace it with the placeholder session."""
        am = self.root.app.authMgr
        am.sessionMgr.delete(req.session)
        req.set_session(self.deletedSession)

    def _finalize_login(self, req: gws.WebRequester, user: gws.User):
        """Replace the current session with a new session for the user."""
        self._delete_session(req)
        am = self.root.app.authMgr
        req.set_session(am.sessionMgr.create(self, user))
        gws.log.info(f'LOGGED_IN: {user.uid=} {user.roles=}')

    def _redirect_target(self, s: str | None) -> str:
        """Convert a redirect target sent by the client to a local URL path."""

        s = (s or '').strip()
        if not s:
            return ''
        if '\\' in s or any(c < ' ' or c == '\x7f' for c in s):
            return ''
        return '/' + s.lstrip('/')

    ##

    def _mfa_start(self, req: gws.WebRequester, user: gws.User) -> gws.AuthMultiFactorTransaction:
        """Start a multi-factor transaction and keep it in a new guest session."""
        am = self.root.app.authMgr

        adapter = am.get_multi_factor_adapter(user.mfaUid)
        if not adapter:
            raise gws.ForbiddenError(f'MFA: {user.mfaUid=} unknown')

        mfa = adapter.start(user)
        if not mfa:
            raise gws.ForbiddenError(f'MFA: {user.mfaUid=} start failed')

        req.set_session(am.sessionMgr.create(self, am.guestUser))

        self._mfa_store(req, mfa)
        return mfa

    def _mfa_verify(self, req: gws.WebRequester, payload: dict) -> gws.AuthMultiFactorTransaction:
        """Verify a payload against the transaction stored in the session."""
        mfa = self._mfa_load(req)
        mfa = mfa.adapter.verify(mfa, payload)

        self._mfa_store(req, mfa)
        return mfa

    def _mfa_restart(self, req: gws.WebRequester) -> gws.AuthMultiFactorTransaction:
        """Restart the transaction stored in the session."""
        mfa = self._mfa_load(req)
        mfa = mfa.adapter.restart(mfa)
        if not mfa:
            raise gws.ForbiddenError(f'MFA: restart failed')

        self._mfa_store(req, mfa)
        return mfa

    def _mfa_store(self, req: gws.WebRequester, mfa: gws.AuthMultiFactorTransaction):
        """Store the transaction in the session."""
        am = self.root.app.authMgr

        sess_mfa = gws.u.merge({}, mfa)
        sess_mfa['user'] = am.serialize_user(mfa.user)
        sess_mfa['adapter'] = mfa.adapter.uid
        req.session.set('AuthMultiFactorTransaction', sess_mfa)

    def _mfa_load(self, req: gws.WebRequester) -> gws.AuthMultiFactorTransaction:
        """Load the transaction from the session and check its state."""
        am = self.root.app.authMgr

        sess_mfa = req.session.get('AuthMultiFactorTransaction')
        if not sess_mfa:
            raise gws.ForbiddenError(f'MFA: transaction not found')

        mfa = gws.AuthMultiFactorTransaction(sess_mfa)
        mfa.adapter = gws.u.require(am.get_multi_factor_adapter(sess_mfa['adapter']))
        mfa.user = gws.u.require(am.unserialize_user(sess_mfa['user']))

        if not mfa.adapter.check_state(mfa):
            raise gws.ForbiddenError(f'MFA: invalid transaction in session')

        return mfa

    def _mfa_response(self, mfa: gws.AuthMultiFactorTransaction, redirect_to: str = '') -> LoginResponse:
        """Create a login response from a multi-factor transaction."""
        return LoginResponse(
            mfaState=mfa.state,
            mfaMessage=mfa.message,
            mfaCanRestart=mfa.adapter.check_restart(mfa),
            redirectTo=redirect_to,
        )
