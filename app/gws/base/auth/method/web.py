import gws
import gws.types as t

from .. import core


@gws.ext.Config('auth.method.web')
class Config(core.MethodConfig):
    """Web-based authorization options"""

    cookieName: str = 'auth'  #: name for the cookie
    cookiePath: str = '/'  #: cookie path


_KIND_STORED = 'web:stored'
_KIND_DELETED = 'web:deleted'


@gws.ext.Object('auth.method.web')
class Object(core.Method):
    cookie_name: str
    cookie_path: str

    def configure(self):
        self.cookie_name = self.var('cookieName', default='auth')
        self.cookie_path = self.var('cookiePath', default='/')

    def open_session(self, auth: core.Manager, req: gws.IWebRequest):
        if self.secure and not req.is_secure:
            return

        sid = req.cookie(self.cookie_name)
        if not sid:
            return

        sess = auth.find_stored_session(sid)
        if not sess or sess.kind != _KIND_STORED:
            gws.log.debug(f'sid={sid} not found')
            return auth.new_session(
                kind=_KIND_DELETED,
                user=auth.guest_user,
                method=self
            )

        return sess

    def close_session(self, auth: core.Manager, sess: core.Session, req: gws.IWebRequest, res: gws.IWebResponse):
        if sess.kind == _KIND_DELETED:
            gws.log.debug('session cookie=deleted')
            res.delete_cookie(
                self.cookie_name,
                path=self.cookie_path)
            return

        if sess.kind == _KIND_STORED:
            gws.log.debug('session cookie=', sess.uid)
            res.set_cookie(
                self.cookie_name,
                value=sess.uid,
                path=self.cookie_path,
                secure=self.secure,
                httponly=True,
            )

            auth.save_stored_session(sess)

    def login(self, auth: core.Manager, credentials: gws.Data, req: gws.IWebRequest) -> t.Optional[core.Session]:
        user = auth.authenticate(self, credentials)
        if user:
            return auth.create_stored_session(_KIND_STORED, self, user)

    def logout(self, auth: core.Manager, sess: core.Session, req: gws.IWebRequest) -> core.Session:
        if sess.kind == _KIND_STORED:
            auth.destroy_stored_session(sess)
        return auth.new_session(
            kind=_KIND_DELETED,
            user=auth.guest_user,
            method=self
        )
