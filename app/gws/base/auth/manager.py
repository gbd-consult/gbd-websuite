"""Authorization and session manager."""

import gws
import gws.config
import gws.lib.date
import gws.lib.json2
import gws.base.web.wsgi
import gws.base.web.error
import gws.types as t

from . import session
from . import user as user_api
from .stores import sqlite

SQLITE_STORE_PATH = gws.MISC_DIR + '/sessions8.sqlite'


class Config(gws.Config):
    """Authentication and authorization options"""

    methods: t.Optional[t.List[gws.ext.config.authMethod]]  #: authorization methods
    providers: t.Optional[t.List[gws.ext.config.authProvider]]  #: authorization providers
    sessionLifeTime: gws.Duration = '1200'  #: session life time
    sessionStore: str = 'sqlite'  #: session storage engine
    sessionStorePath: t.Optional[str]  #: session storage path


class Object(gws.Node, gws.IAuthManager):
    """Authorization manager."""

    sessionLifeTime: int
    store: sqlite.SessionStore
    guestSession: session.Session
    webMethod: gws.IAuthMethod

    def configure(self):

        self.sessionLifeTime = self.var('sessionLifeTime')

        if self.var('sessionStore') == 'sqlite':
            self.store = sqlite.SessionStore(self.var('sessionStorePath', default=SQLITE_STORE_PATH))
        else:
            # @TODO other store types
            raise gws.ConfigurationError('invalid session store type')

        self.providers = self.create_children(gws.ext.object.authProvider, self.var('providers'))
        sys_provider = self.create_child(gws.ext.object.authProvider, {'type': 'system'})
        self.providers.append(sys_provider)

        for p in self.providers:
            p.auth = self

        self.guestUser = sys_provider.get_user('guest')
        self.systemUser = sys_provider.get_user('system')

        self.guestSession = session.Session('guest_session', method=None, user=self.guestUser)

        self.methods = self.create_children(gws.ext.object.authMethod, self.var('methods'))
        # if no methods configured, enable the Web method
        if not self.methods:
            self.methods.append(self.create_child(gws.ext.object.authMethod, {'type': 'web'}))

        for p in self.methods:
            p.auth = self

        self.webMethod = self.get_method(ext_type='web')

    ##

    def activate(self):
        self.root.app.register_web_middleware('auth', self.auth_middleware)

    def auth_middleware(self, req: gws.IWebRequester, nxt) -> gws.IWebResponder:
        req.session = self._open_session(req)
        req.user = req.session.user
        gws.log.debug(f'auth_open: typ={req.session.typ!r} user={req.user.uid!r}')
        res = nxt()
        sess = getattr(req, 'session')
        gws.log.debug(f'auth_close: typ={sess.typ!r} user={sess.user.uid!r}')
        req.session = self._close_session(sess, req, res)
        return res

    def _open_session(self, req):
        for m in self.methods:
            sess = m.open_session(req)
            if sess:
                return sess
        return self.guestSession

    def _close_session(self, sess: gws.IAuthSession, req, res):
        if sess and sess.method:
            return sess.method.close_session(sess, req, res)
        return self.guestSession

    def web_login(self, req, credentials):
        if not req.user.isGuest:
            gws.log.error('login while logged-in')
            raise gws.base.web.error.Forbidden()

        if not self.webMethod:
            raise gws.base.web.error.Forbidden()

        sess = self.webMethod.login(req, credentials)
        if not sess:
            raise gws.base.web.error.Forbidden()

        req.session = sess
        req.user = req.session.user
        gws.log.debug(f'auth_login: typ={req.session.typ!r} user={req.user.uid!r}')

    def web_logout(self, req):
        if req.user.isGuest:
            return

        if not self.webMethod:
            return

        sess = getattr(req, 'session')
        if sess.method != self.webMethod:
            gws.log.error(f'wrong method for logout: {sess.method!r}')
            raise gws.base.web.error.Forbidden()

        req.session = self.webMethod.logout(req, sess)
        req.user = req.session.user

        gws.log.debug(f'auth_logout: typ={sess.typ!r}')

    ##

    def authenticate(self, method, credentials):
        for prov in self.providers:
            if prov.allowedMethods and method.extType not in prov.allowedMethods:
                continue
            gws.log.debug(f'trying provider {prov.uid!r}')
            user = prov.authenticate(method, credentials)
            if user:
                return user

    ##

    def get_user(self, user_uid):
        provider_uid, local_uid = user_api.parse_uid(user_uid)
        prov = self.get_provider(provider_uid)
        if prov:
            return prov.get_user(local_uid)

    def get_provider(self, uid=None, ext_type=None):
        for obj in self.providers:
            if (uid and obj.uid == uid) or (ext_type and obj.extType == ext_type):
                return obj

    def get_method(self, uid=None, ext_type=None):
        for obj in self.methods:
            if (uid and obj.uid == uid) or (ext_type and obj.extType == ext_type):
                return obj

    def serialize_user(self, user):
        return gws.lib.json2.to_string([user.provider.uid, user.provider.serialize_user(user)])

    def unserialize_user(self, ser):
        provider_uid, str_user = gws.lib.json2.from_string(ser)
        prov = self.get_provider(provider_uid)
        return prov.unserialize_user(str_user) if prov else None

    ##

    def session_find(self, uid):
        self.store.cleanup(self.sessionLifeTime)

        rec = self.store.find(uid)
        if not rec:
            return None

        age = gws.lib.date.timestamp() - rec['updated']
        if age > self.sessionLifeTime:
            gws.log.debug(f'sess uid={uid!r} EXPIRED age={age!r}')
            self.store.delete(uid)
            return None

        user = self.unserialize_user(rec['str_user'])
        if not user:
            gws.log.error(f'FAILED to unserialize user from sess={uid!r}')
            self.store.delete(uid)
            return None

        return session.Session(
            rec['typ'],
            uid=rec['uid'],
            method=self.get_method(rec['method_uid']),
            user=user,
            data=gws.lib.json2.from_string(rec['str_data']),
            saved=True
        )

    def session_create(self, typ: str, method: gws.IAuthMethod, user: gws.IUser) -> gws.IAuthSession:
        uid = gws.random_string(64)
        return session.Session(typ, user, method, uid)

    def session_save(self, sess: gws.IAuthSession):
        if not sess.saved:
            self.store.create(
                uid=sess.uid,
                typ=sess.typ,
                method_uid=sess.method.uid,
                provider_uid=sess.user.provider.uid,
                user_uid=sess.user.uid,
                str_user=self.serialize_user(sess.user))
            sess.saved = True
        elif sess.changed:
            self.store.update(sess.uid, str_data=gws.lib.json2.to_string(sess.data))
        else:
            self.store.touch(sess.uid)

    def session_delete(self, sess: gws.IAuthSession):
        self.store.delete(sess.uid)

    def session_delete_all(self):
        self.store.delete_all()

    def stored_session_records(self) -> t.List[dict]:
        return self.store.get_all()
