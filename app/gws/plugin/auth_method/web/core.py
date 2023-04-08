"""Web authorisation method."""

import gws
import gws.base.auth.method
import gws.base.web.error

gws.ext.new.authMethod('web')


class Config(gws.base.auth.method.Config):
    """Web-based authorization options"""

    cookieName: str = 'auth'
    """name for the cookie"""
    cookiePath: str = '/'
    """cookie path"""


_ACTIVE = 'web:active'
_DELETED = 'web:deleted'


class Object(gws.base.auth.method.Object):
    cookieName: str
    cookiePath: str
    deletedSession: gws.IAuthSession

    def configure(self):
        self.cookieName = self.cfg('cookieName')
        self.cookiePath = self.cfg('cookiePath')

    def activate(self):
        self.deletedSession = self.authMgr.session_create(_DELETED, user=self.authMgr.guestUser, method=self)

    def open_session(self, req):
        sid = req.cookie(self.cookieName)
        if not sid:
            return False

        if self.secure and not req.isSecure:
            gws.log.debug(f'insecure context, session {sid!r} ignored')
            return False

        sess = self.authMgr.session_find(sid)

        if not sess or sess.typ != _ACTIVE:
            gws.log.debug(f'open_session: sid={sid} not found or invalid')
            self.authMgr.session_delete(sess)
            self.authMgr.session_activate(req, self.deletedSession)
            return True

        if not self.mfa_check(sess):
            gws.log.debug(f'open_session: sid={sid} mfa check failed')
            self.authMgr.session_delete(sess)
            self.authMgr.session_activate(req, self.deletedSession)
            return True

        self.authMgr.session_activate(req, sess)
        gws.log.debug(f'open_session: sid={sess.uid!r} user={sess.user.uid!r}')

        gws.p(sess.user.roles)
        gws.p(sess.user.pendingMfa)
        return True

    def close_session(self, req, res):
        sess = getattr(req, 'session')
        if not sess:
            return False

        if sess.typ == _DELETED:
            gws.log.debug('session cookie=deleted')
            res.delete_cookie(
                self.cookieName,
                path=self.cookiePath)

            self.authMgr.session_activate(req, None)
            return True

        if sess.typ == _ACTIVE:
            if res.status < 400:
                gws.log.debug(f'session cookie={sess.uid!r}')
                res.set_cookie(
                    self.cookieName,
                    value=sess.uid,
                    path=self.cookiePath,
                    secure=self.secure,
                    httponly=True)
                self.authMgr.session_save(sess)

            self.authMgr.session_activate(req, None)
            return True

    def action_login(self, req: gws.IWebRequester, credentials: gws.Data):
        if not req.user.isGuest:
            gws.log.error('login while logged-in')
            raise gws.base.web.error.Forbidden()

        if self.secure and not req.isSecure:
            gws.log.debug(f'insecure context, login failed')
            raise gws.base.web.error.Forbidden()

        user = self.authMgr.authenticate(self, credentials)
        if not user:
            raise gws.base.web.error.Forbidden()

        sess = self.authMgr.session_create(_ACTIVE, self, user)
        self.authMgr.session_activate(req, sess)

        self.mfa_start(sess)

        gws.log.info(f'LOGGED_IN: sess={sess.typ!r} user={sess.user.uid!r}')

    def action_logout(self, req: gws.IWebRequester):
        sess = getattr(req, 'session')
        if not sess or sess.user.isGuest:
            return

        if sess.method != self:
            gws.log.error(f'wrong method for logout: {sess.method!r}')
            raise gws.base.web.error.Forbidden()

        gws.log.info(f'LOGGED_OUT: typ={sess.typ!r} user={sess.user.uid!r}')

        self.mfa_cancel(sess)
        self.authMgr.session_delete(sess)
        self.authMgr.session_activate(req, self.deletedSession)

    def action_mfa_verify(self, req: gws.IWebRequester, request: gws.Data):
        sess = getattr(req, 'session')
        if not sess:
            return

        if sess.method != self:
            gws.log.error(f'wrong method for mfa_verify method={sess.method!r}')
            raise gws.base.web.error.Forbidden()

        self.mfa_verify(sess, request)

        new_sess = self.authMgr.session_create(_ACTIVE, self, sess.user)
        self.authMgr.session_delete(sess)
        self.authMgr.session_activate(req, new_sess)

    ##

    def mfa_check(self, sess):
        if not sess.user.pendingMfa:
            return True
        mfa = self.authMgr.get_mfa(sess.user.pendingMfa.methodUid)
        return mfa.is_valid(sess.user)

    def mfa_start(self, sess):
        if not sess.user.pendingMfa:
            return

        mfa = self.authMgr.get_mfa(sess.user.pendingMfa.methodUid)
        mfa.start(sess.user)

    def mfa_cancel(self, sess):
        if not sess.user.pendingMfa:
            return

        mfa = self.authMgr.get_mfa(sess.user.pendingMfa.methodUid)
        mfa.cancel(sess.user)

    def mfa_verify(self, sess: gws.IAuthSession, request):
        if not sess.user.pendingMfa:
            gws.log.error(f'pendingMfa missing, user={sess.user.uid!r}')
            raise gws.base.web.error.Forbidden()

        mfa = self.authMgr.get_mfa(sess.user.pendingMfa.methodUid)
        ok = mfa.verify(sess.user, request)

        if not ok:
            gws.log.error(f'mfa verify failed user={sess.user.uid!r} request={request!r}')
            raise gws.base.web.error.Forbidden()

        gws.log.info(f'LOGGED_IN_AND_VERIFIED: user={sess.user.uid!r} mfa={mfa.uid}')

        sess.user.pendingMfa = None
