import werkzeug.wrappers

import gws
import gws.auth.session
import gws.auth.api
import gws.types as t
from . import wrappers, error

# @TODO skip updates if heartBeat is enabled

_DELETED = object()


#:stub
class WebRequest(wrappers.BaseWebRequest):
    def __init__(self, root, environ, site):
        super().__init__(root, environ, site)

        self._session = None
        self._user = None

        self.header_name = None
        s = self.root.var('auth.header')
        if s:
            self.header_name = 'HTTP_' + s.upper().replace('-', '_')

        self.cookie_name = self.root.var('auth.cookie.name')

    @property
    def user(self) -> t.User:
        if not self._user:
            raise ValueError('auth_begin not called')
        return self._user

    def require(self, klass: str, uid: str) -> t.Object:
        node = self.root.find(klass, uid)
        if not node:
            gws.log.error('require: not found', klass, uid)
            raise error.NotFound()
        if not self.user.can_use(node):
            gws.log.error('require: denied', klass, uid)
            raise error.Forbidden()
        return node

    def require_project(self, uid: str) -> t.ProjectObject:
        p: t.ProjectObject = self.require('gws.common.project', uid)
        return p

    def acquire(self, klass: str, uid: str) -> t.Object:
        node = self.root.find(klass, uid)
        if node and self.user.can_use(node):
            return node

    def login(self, username: str, password: str):
        if not self.user.is_guest:
            gws.log.error('login while logged-in')
            raise gws.web.error.Forbidden()

        try:
            user = gws.auth.api.authenticate_user(username, password)
        except gws.auth.api.Error as err:
            raise error.Forbidden() from err

        if not user:
            raise error.Forbidden()

        self._stop_session()
        self._session = self._session_manager.create_for(user)
        self._user = user

    def logout(self):
        self._stop_session()
        self._user = self._guest_user

    def auth_begin(self):
        self._session = self._init_session()
        if self._session and self._session is not _DELETED:
            self._user = self._session.user
        else:
            self._user = self._guest_user

    def auth_commit(self, res):
        if not self._session:
            return

        if self._session is _DELETED:
            if self.cookie_name:
                gws.log.info('session cookie=deleted')
                res.delete_cookie(self.cookie_name)
            return

        self._session_manager.update(self._session)

        if self.cookie_name:
            gws.log.info('session cookie=', self._session.uid)
            res.set_cookie(
                self.cookie_name,
                value=self._session.uid,
                **self._cookie_options
            )

    def _init_session(self):
        sid = self._session_id
        if not sid:
            return

        gws.log.debug('found sid', sid)
        sess = self._session_manager.find(sid)
        if not sess:
            gws.log.info('no session for sid', sid)
            return _DELETED

        return sess

    @property
    def _session_id(self):
        # @TODO secure

        if self.header_name in self.environ:
            return self.environ[self.header_name]

        if self.cookie_name in self.cookies:
            return self.cookies[self.cookie_name]

    def _stop_session(self):
        if self._session and self._session is not _DELETED:
            self._session_manager.delete(self._session)
            self._session = _DELETED
        else:
            self._session = None

    @property
    def _cookie_options(self):
        d = {
            'path': self.root.var('auth.session.cookie.path', default='/'),
            'httponly': True
        }
        # domain=str(self.root.get('auth.session.cookie.domain')),
        # secure=self.root.get('auth.session.httpsOnly'),
        return d

    @property
    def _session_manager(self):
        return gws.get_global('auth.session_manager', gws.auth.session.Manager)

    @property
    def _guest_user(self):
        def get():
            p: t.AuthProviderObject = self.root.find_first('gws.ext.auth.provider.system')
            return p.get_user('guest')

        return gws.get_global('auth.guest_user', get)
