import gws.common.auth.method
import gws.common.auth.error
import gws.common.auth.mfa
import gws.web.error

import gws.types as t


class Config(t.WithType):
    """Web-based authorization options"""

    cookieName: str = 'auth'  #: name for the cookie
    cookiePath: str = '/'  #: cookie path
    secure: bool = True  #: use only with SSL


_ST_STORED = 'web:stored'
_ST_DELETED = 'web:deleted'


class Object(gws.common.auth.method.Object):

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

    ##

    def action_login(self, req: t.IRequest, username: str, password: str):
        if not req.user.is_guest:
            gws.log.error('login while logged-in')
            raise gws.web.error.Forbidden()

        try:
            user = self.auth.authenticate(self, username, password)
            if not user:
                raise gws.common.auth.error.LoginNotFound()
        except gws.common.auth.error.Error as exc:
            raise gws.web.error.Forbidden() from exc

        if user.attribute('mfauid'):
            req.session = self._create_mfa_session(user)
        else:
            req.session = self.auth.create_stored_session(_ST_STORED, self, user)

        return self.action_response(req)

    def action_logout(self, req: t.IRequest):
        self._destroy_and_logout(req)
        return self.action_response(req)

    def action_mfa_verify(self, req: t.IRequest, data: t.Data):
        mfa_obj, user, mf = self._require_pending_mfa(req)

        try:
            ok = mfa_obj.verify(user, mf, data)
        except gws.common.auth.mfa.Error:
            gws.log.exception()
            self._destroy_and_logout(req)
            raise gws.web.error.Forbidden()

        gws.log.debug(f'MFA: verify, result={ok} user={user.uid!r}')

        if ok:
            self.auth.destroy_stored_session(req.session)
            req.session = self.auth.create_stored_session(_ST_STORED, self, user)
            return self.action_response(req)

        req.session.set('pendingMfa', gws.as_dict(mf))
        self.auth.save_stored_session(req.session)

        raise gws.web.error.Conflict()

    def action_mfa_restart(self, req: t.IRequest):
        mfa_obj, user, mf = self._require_pending_mfa(req)

        try:
            mfa_obj.restart(user, mf)
        except gws.common.auth.mfa.Error:
            gws.log.exception()
            self._destroy_and_logout(req)
            raise gws.web.error.Forbidden()

        gws.log.debug(f'MFA: restart, user={user.uid!r}')

        req.session.set('pendingMfa', gws.as_dict(mf))
        self.auth.save_stored_session(req.session)

        return self.action_response(req)

    def action_response(self, req: t.IRequest):
        res = t.Data(user=req.user.props)
        pm = self._get_pending_mfa(req)
        if pm:
            res.mfaOptions = pm[2].clientOptions
        return res

    ##

    def _create_mfa_session(self, user):
        mfa_uid = user.attribute('mfauid')
        mfa_obj = self.auth.get_mfa(mfa_uid)
        if not mfa_obj:
            gws.log.error(f'MFA: not found, uid={mfa_uid!r}')
            raise gws.web.error.Forbidden()

        try:
            mf = mfa_obj.start(user)
        except gws.common.auth.mfa.Error:
            gws.log.exception()
            raise gws.web.error.Forbidden()

        sess = self.auth.create_stored_session(_ST_STORED, self, self.auth.guest_user)
        sess.set('pendingMfa', gws.as_dict(mf))
        sess.set('pendingMfaUser', self.auth.serialize_user(user))
        return sess

    def _require_pending_mfa(self, req: t.IRequest):
        pm = self._get_pending_mfa(req)
        if not pm:
            raise gws.web.error.Forbidden()
        return pm

    def _get_pending_mfa(self, req: t.IRequest):
        d = req.session.get('pendingMfa')
        if not d:
            return

        mf = t.AuthMfaData(d)

        mfa_obj = self.auth.get_mfa(mf.uid)
        if not mfa_obj:
            gws.log.error(f'MFA: not found, uid={mf.uid!r}')
            self._destroy_and_logout(req)
            return

        u = req.session.get('pendingMfaUser')
        if not u:
            gws.log.debug(f'MFA: no user, uid={mf.uid!r}')
            self._destroy_and_logout(req)
            return

        user = self.auth.unserialize_user(u)

        if not mfa_obj.is_valid(user, mf):
            gws.log.debug(f'MFA: invalid, uid={mf.uid!r}')
            self._destroy_and_logout(req)
            return

        return mfa_obj, user, mf

    def _destroy_and_logout(self, req):
        if req.session.type == _ST_STORED:
            self.auth.destroy_stored_session(req.session)
        req.session = self.auth.new_session(
            type=_ST_DELETED,
            user=self.auth.guest_user,
            method=self
        )
