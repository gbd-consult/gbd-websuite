import gws
import gws.auth.session as session
import gws.config
import gws.types as t
from . import wrappers, error

# @TODO skip updates if heartBeat is enabled

DELETED = object()


def _session_manager():
    return gws.get_global('auth.session_manager', session.Manager)


def _get_guest_user():
    p: t.AuthProviderInterface = gws.config.find('gws.ext.auth.provider', 'system')
    return p.get_user('guest')


def _guest_user():
    return gws.get_global('auth.guest_user', _get_guest_user)


class AuthRequest(wrappers.Request):
    _session = None
    _user = None

    @property
    def user(self):
        if not self._user:
            raise ValueError('no auth!')
        return self._user

    def require(self, klass, uid):
        node = gws.config.find(klass, uid)
        if not node:
            gws.log.error('require: not found', klass, uid)
            raise error.NotFound()
        if not self.user.can_use(node):
            gws.log.error('require: denied', klass, uid)
            raise error.Forbidden()
        return node

    def require_project(self, uid):
        return self.require('gws.common.project', uid)

    def acquire(self, klass, uid):
        node = gws.config.find(klass, uid)
        if node and self.user.can_use(node):
            return node

    def logged_in(self, user):
        self._stop_session()
        self._session = _session_manager().create_for(user)
        self._user = user

    def logged_out(self):
        self._stop_session()
        self._user = _guest_user()

    def _session_id(self):
        # @TODO secure

        if gws.config.var('auth.header'):
            s = 'HTTP_' + gws.config.var('auth.header').upper().replace('-', '_')
            if s in self.environ:
                return self.environ[s]

        if gws.config.var('auth.cookie'):
            s = gws.config.var('auth.cookie.name')
            if s in self.cookies:
                return self.cookies[s]

    def auth_begin(self):
        self._session = self._init_session()
        if self._session and self._session is not DELETED:
            self._user = self._session.user
        else:
            self._user = _guest_user()

    def auth_commit(self, res):
        if not self._session:
            return

        cookie = None
        if gws.config.var('auth.cookie'):
            cookie = gws.config.var('auth.cookie.name')

        if self._session is DELETED:
            if cookie:
                gws.log.info('session cookie=deleted')
                res.delete_cookie(cookie)
            return

        _session_manager().update(self._session)

        if cookie:
            gws.log.info('session cookie=', self._session.uid)
            res.set_cookie(
                cookie,
                value=self._session.uid,
                **self._cookie_options()
            )

    def _init_session(self):
        sid = self._session_id()
        if not sid:
            return

        gws.log.debug('found sid', sid)
        sess = _session_manager().find(sid)
        if not sess:
            gws.log.info('no session for sid', sid)
            return DELETED

        return sess

    def _stop_session(self):
        if self._session and self._session is not DELETED:
            _session_manager().delete(self._session)
            self._session = DELETED
        else:
            self._session = None

    def _cookie_options(self):
        d = {'path': gws.config.var('auth.session.cookie.path', default='/'), 'httponly': True}
        # domain=str(gws.config.get('auth.session.cookie.domain')),
        # secure=gws.config.get('auth.session.httpsOnly'),
        return d
