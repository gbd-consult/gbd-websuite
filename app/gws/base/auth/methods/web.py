import gws
import gws.types as t

from .. import method


@gws.ext.config.authMethod('web')
class Config(method.Config):
    """Web-based authorization options"""

    cookieName: str = 'auth'  #: name for the cookie
    cookiePath: str = '/'  #: cookie path


_ACTIVE = 'web:active'
_DELETED = 'web:deleted'


@gws.ext.object.authMethod('web')
class Object(method.Object):
    cookieName: str
    cookiePath: str
    deletedSession: gws.IAuthSession

    def configure(self):
        super().configure()

        self.cookieName = self.var('cookieName')
        self.cookiePath = self.var('cookiePath')

    def activate(self):
        self.deletedSession = self.auth.session_create(_DELETED, user=self.auth.guestUser, method=self)

    def open_session(self, req):
        sid = req.cookie(self.cookieName)
        if not sid:
            return

        if self.secure and not req.isSecure:
            gws.log.debug(f'insecure context, session {sid!r} ignored')
            return

        sess = self.auth.session_find(sid)
        if sess and sess.typ == _ACTIVE:
            return sess

        gws.log.debug(f'sid={sid} not found or invalid')
        return self.deletedSession

    def close_session(self, sess, req, res):
        if sess.typ == _DELETED:
            gws.log.debug('session cookie=deleted')
            res.delete_cookie(
                self.cookieName,
                path=self.cookiePath)

        if sess.typ == _ACTIVE and res.status < 400:
            gws.log.debug('session cookie=', sess.uid)
            res.set_cookie(
                self.cookieName,
                value=sess.uid,
                path=self.cookiePath,
                secure=self.secure,
                httponly=True,
            )
            self.auth.session_save(sess)

    def login(self, req, credentials):
        user = self.auth.authenticate(self, credentials)
        if user:
            return self.auth.session_create(_ACTIVE, self, user)

    def logout(self, req, sess):
        if sess.typ == _ACTIVE:
            self.auth.session_delete(sess)
        return self.deletedSession
