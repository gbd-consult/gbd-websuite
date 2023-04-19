"""Web authorisation method."""

import gws
import gws.base.auth
import gws.base.web.error

gws.ext.new.authMethod('web')


class Config(gws.base.auth.method.Config):
    """Web-based authorization options"""

    cookieName: str = 'auth'
    """name for the cookie"""
    cookiePath: str = '/'
    """cookie path"""


_DELETED_SESSION = 'web:deleted'


class Object(gws.base.auth.method.Object):
    cookieName: str
    cookiePath: str

    deletedSession: gws.base.auth.session.Object

    def configure(self):
        self.cookieName = self.cfg('cookieName', default=Config.cookieName)
        self.cookiePath = self.cfg('cookiePath', default=Config.cookiePath)

    def activate(self):
        self.deletedSession = gws.base.auth.session.Object(
            uid=_DELETED_SESSION,
            method=self,
            user=self.authMgr.guestUser,
        )

    def open_session(self, req):
        sid = req.cookie(self.cookieName)
        if not sid:
            return

        if self.secure and not req.isSecure:
            gws.log.debug(f'insecure context, {sid=} ignored')
            return

        sess = self.authMgr.sessionMgr.get_valid(sid)

        if not sess:
            gws.log.debug(f'open_session: {sid=} not found or invalid')
            return self.deletedSession

        return sess

    def close_session(self, req, res):
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
                value=sess.uid,
                path=self.cookiePath,
                secure=self.secure,
                httponly=True)
            self.authMgr.sessionMgr.save(sess)

    def handle_login(self, req: gws.IWebRequester, credentials: gws.Data):
        if not req.user.isGuest:
            gws.log.error('login while logged-in')
            raise gws.base.web.error.Forbidden()

        if self.secure and not req.isSecure:
            gws.log.debug(f'insecure context, login failed')
            raise gws.base.web.error.Forbidden()

        user = self.authMgr.authenticate(self, credentials)
        if not user:
            raise gws.base.web.error.Forbidden()

        sess = self.authMgr.sessionMgr.create(self, user)
        req.set_session(sess)

        gws.log.info(f'LOGGED_IN: user={req.session.user.uid!r}')

    def handle_logout(self, req: gws.IWebRequester):
        if req.user.isGuest:
            return

        sess = req.session

        if req.session.method != self:
            gws.log.error(f'wrong method for logout: {sess.method!r}')
            raise gws.base.web.error.Forbidden()

        gws.log.info(f'LOGGED_OUT: user={sess.user.uid!r}')

        self.authMgr.sessionMgr.delete(sess)
        req.set_session(self.deletedSession)
