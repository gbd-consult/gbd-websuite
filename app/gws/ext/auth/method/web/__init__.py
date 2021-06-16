import gws.base.auth.method
import gws.base.auth.error

import gws.types as t


class Config(t.WithType):
    """Web-based authorization options"""

    cookieName: str = 'auth'  #: name for the cookie
    cookiePath: str = '/'  #: cookie path
    secure: bool = True  #: use only with SSL


_ST_STORED = 'web:stored'
_ST_DELETED = 'web:deleted'


class Object(gws.base.auth.method.Object):

    def configure(self):
        super().configure()
        self.cookie_name = self.var('cookieName', default='auth')
        self.cookie_path = self.var('cookiePath', default='/')
        self.secure = self.var('secure')

    def open_session(self, auth: t.IAuthManager, req: t.IRequest):
        if self.secure and not req.is_secure:
            return

        sid = req.cookie(self.cookie_name)
        if not sid:
            return

        sess = auth.find_stored_session(sid)
        if not sess or sess.type != _ST_STORED:
            gws.log.debug(f'sid={sid} not found')
            return auth.new_session(
                type=_ST_DELETED,
                user=auth.guest_user,
                method=self
            )

        return sess

    def close_session(self, auth: t.IAuthManager, sess: t.ISession, req: t.IRequest, res: t.IResponse):
        if sess.type == _ST_DELETED:
            gws.log.debug('session cookie=deleted')
            res.delete_cookie(
                self.cookie_name,
                path=self.cookie_path)
            return

        if sess.type == _ST_STORED:
            gws.log.debug('session cookie=', sess.uid)
            res.set_cookie(
                self.cookie_name,
                value=sess.uid,
                path=self.cookie_path,
                secure=self.secure,
                httponly=True,
            )

            auth.save_stored_session(sess)

    def login(self, auth: t.IAuthManager, username: str, password: str, req: t.IRequest) -> t.Optional[t.ISession]:
        user = auth.authenticate(self, username, password)
        if user:
            return auth.create_stored_session(_ST_STORED, self, user)

    def logout(self, auth: t.IAuthManager, sess: t.ISession, req: t.IRequest) -> t.ISession:
        if sess.type == _ST_STORED:
            auth.destroy_stored_session(sess)
        return auth.new_session(
            type=_ST_DELETED,
            user=auth.guest_user,
            method=self
        )
