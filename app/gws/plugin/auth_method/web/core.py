"""Web authorisation method."""

from typing import Optional, cast

import gws
import gws.base.auth
import gws.base.web

gws.ext.new.authMethod('web')


class Config(gws.base.auth.method.Config):
    """Web-based authorization options"""

    cookieName: str = 'auth'
    """name for the cookie"""
    cookiePath: str = '/'
    """cookie path"""


##

class UserResponse(gws.Response):
    user: Optional[gws.base.auth.user.Props]


class LogoutResponse(gws.Response):
    pass


class LoginRequest(gws.Request):
    username: str
    password: str


class LoginResponse(gws.Response):
    user: Optional[gws.base.auth.user.Props]
    mfaState: Optional[gws.AuthMultiFactorState]
    mfaMessage: str = ''
    mfaCanRestart: bool = False


class MfaVerifyRequest(gws.Request):
    payload: dict


##

_DELETED_SESSION = 'web:deleted'


class Object(gws.base.auth.method.Object):
    cookieName: str
    cookiePath: str

    deletedSession: gws.base.auth.session.Object

    def configure(self):
        self.uid = 'gws.plugin.self.web'
        self.cookieName = self.cfg('cookieName', default=Config.cookieName)
        self.cookiePath = self.cfg('cookiePath', default=Config.cookiePath)

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

        sess = am.sessionMgr.get_valid(sid)

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
                path=self.cookiePath)
            return

        if res.status < 400:
            gws.log.debug(f'session cookie={sess.uid!r}')
            res.set_cookie(
                self.cookieName,
                sess.uid,
                path=self.cookiePath,
                secure=self.secure,
                httponly=True)
            am.sessionMgr.save(sess)

    def handle_login(self, req: gws.WebRequester, p: LoginRequest) -> LoginResponse:
        if not req.user.isGuest:
            raise gws.ForbiddenError(f'login: already logged-in {req.user.uid=}')

        if self.secure and not req.isSecure:
            raise gws.ForbiddenError('login: insecure_context, ignored')

        user = self.root.app.authMgr.authenticate(self, p)
        if not user:
            raise gws.ForbiddenError('login: user not found')

        if user.mfaUid:
            mfa = self._mfa_start(req, user)
            gws.log.info(f'LOGGED_IN (MFA pending): {user.uid=} {user.roles=}')
            return self._mfa_response(mfa)

        self._finalize_login(req, user)
        return LoginResponse(user=gws.props_of(user, user))

    def handle_mfa_verify(self, req: gws.WebRequester, p: MfaVerifyRequest) -> LoginResponse:
        try:
            mfa = self._mfa_verify(req, p.payload)
        except gws.ForbiddenError:
            self._delete_session(req)
            raise

        if mfa.state == gws.AuthMultiFactorState.ok:
            self._finalize_login(req, mfa.user)
            return self._mfa_response(mfa)

        if mfa.state == gws.AuthMultiFactorState.retry:
            return self._mfa_response(mfa)

        self._delete_session(req)
        raise gws.ForbiddenError(f'MFA: verify failed {mfa.state=}')

    def handle_mfa_restart(self, req: gws.WebRequester, p: gws.Request) -> LoginResponse:
        try:
            mfa = self._mfa_restart(req)
        except gws.ForbiddenError:
            self._delete_session(req)
            raise

        return self._mfa_response(mfa)

    def handle_logout(self, req: gws.WebRequester) -> LogoutResponse:
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
        am = self.root.app.authMgr
        am.sessionMgr.delete(req.session)
        req.set_session(self.deletedSession)

    def _finalize_login(self, req: gws.WebRequester, user: gws.User):
        self._delete_session(req)
        am = self.root.app.authMgr
        req.set_session(am.sessionMgr.create(self, user))
        gws.log.info(f'LOGGED_IN: {user.uid=} {user.roles=}')

    ##

    def _mfa_start(self, req: gws.WebRequester, user: gws.User) -> gws.AuthMultiFactorTransaction:
        am = self.root.app.authMgr

        adapter = am.get_mf_adapter(user.mfaUid)
        if not adapter:
            raise gws.ForbiddenError(f'MFA: {user.mfaUid=} unknown')

        mfa = adapter.start(user)
        if not mfa:
            raise gws.ForbiddenError(f'MFA: {user.mfaUid=} start failed')

        req.set_session(am.sessionMgr.create(self, am.guestUser))

        self._mfa_store(req, mfa)
        return mfa

    def _mfa_verify(self, req: gws.WebRequester, payload: dict) -> gws.AuthMultiFactorTransaction:
        mfa = self._mfa_load(req)
        mfa = mfa.adapter.verify(mfa, payload)

        self._mfa_store(req, mfa)
        return mfa

    def _mfa_restart(self, req: gws.WebRequester) -> gws.AuthMultiFactorTransaction:
        mfa = self._mfa_load(req)
        mfa = mfa.adapter.restart(mfa)
        if not mfa:
            raise gws.ForbiddenError(f'MFA: restart failed')

        self._mfa_store(req, mfa)
        return mfa

    def _mfa_store(self, req: gws.WebRequester, mfa: gws.AuthMultiFactorTransaction):
        am = self.root.app.authMgr

        sess_mfa = gws.u.merge({}, mfa)
        sess_mfa['user'] = am.serialize_user(mfa.user)
        sess_mfa['adapter'] = mfa.adapter.uid
        req.session.set('AuthMultiFactorTransaction', sess_mfa)

    def _mfa_load(self, req: gws.WebRequester) -> gws.AuthMultiFactorTransaction:
        am = self.root.app.authMgr

        sess_mfa = req.session.get('AuthMultiFactorTransaction')
        if not sess_mfa:
            raise gws.ForbiddenError(f'MFA: transaction not found')

        mfa = gws.AuthMultiFactorTransaction(sess_mfa)
        mfa.adapter = am.get_mf_adapter(sess_mfa['adapter'])
        mfa.user = am.unserialize_user(sess_mfa['user'])

        if not mfa.adapter.check_state(mfa):
            raise gws.ForbiddenError(f'MFA: invalid transaction in session')

        return mfa

    def _mfa_response(self, mfa: gws.AuthMultiFactorTransaction) -> LoginResponse:
        return LoginResponse(
            mfaState=mfa.state,
            mfaMessage=mfa.message,
            mfaCanRestart=mfa.adapter.check_restart(mfa),
        )
