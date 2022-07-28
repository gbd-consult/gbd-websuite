import gws
import gws.types as t

from .. import manager, method


@gws.ext.config.authMethod('web')
class Config(method.Config):
    """Web-based authorization options"""

    cookieName: str = 'auth'  #: name for the cookie
    cookiePath: str = '/'  #: cookie path


_ACTIVE = 'web:active'
_DELETED = 'web:deleted'


@gws.ext.object.authMethod('web')
class Object(method.Object):
    cookie_name: str
    cookie_path: str

    def configure(self):
        self.cookie_name = self.var('cookieName')
        self.cookie_path = self.var('cookiePath')

    def open_session(self, auth, req):
        sid = req.cookie(self.cookie_name)
        if not sid:
            return

        if self.secure and not req.is_secure:
            gws.log.debug(f'insecure context, session {sid!r} ignored')
            return

        sess = t.cast(manager.Object, auth).find_stored_session(sid)
        if sess and sess.typ == _ACTIVE:
            return sess

        gws.log.debug(f'sid={sid} not found or invalid')
        return t.cast(manager.Object, auth).new_session(_DELETED, user=auth.guest_user, method=self)

    def close_session(self, auth, sess, req, res):
        if sess.typ == _DELETED:
            gws.log.debug('session cookie=deleted')
            res.delete_cookie(
                self.cookie_name,
                path=self.cookie_path)

        if sess.typ == _ACTIVE and res.status_code < 400:
            gws.log.debug('session cookie=', sess.uid)
            res.set_cookie(
                self.cookie_name,
                value=sess.uid,
                path=self.cookie_path,
                secure=self.secure,
                httponly=True,
            )
            t.cast(manager.Object, auth).save_stored_session(sess)

    def login(self, auth, credentials, req):
        user = auth.authenticate(self, credentials)
        if user:
            return t.cast(manager.Object, auth).new_stored_session(_ACTIVE, self, user)

    def logout(self, auth, sess, req):
        if sess.typ == _ACTIVE:
            t.cast(manager.Object, auth).destroy_stored_session(sess)
        return t.cast(manager.Object, auth).new_session(_DELETED, auth.guest_user, method=self)
