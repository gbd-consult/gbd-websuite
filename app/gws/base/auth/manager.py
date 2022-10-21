"""Authorization and session manager."""

import gws
import gws.config
import gws.lib.date
import gws.lib.jsonx
import gws.base.web.wsgi
import gws.base.web.error
import gws.types as t

from . import session
from . import user as user_api
from .stores import sqlite

SQLITE_STORE_PATH = gws.MISC_DIR + '/sessions8.sqlite'


class SessionOptions(gws.Data):
    lifeTime: int
    store: sqlite.SessionStore
    storePath: str


class SessionConfig(gws.Config):
    lifeTime: gws.Duration = '1200'  #: session life time
    store: str = 'sqlite'  #: session storage engine
    storePath: t.Optional[str]  #: session storage path


class Config(gws.Config):
    """Authentication and authorization options"""

    methods: t.Optional[t.List[gws.ext.config.authMethod]]  #: authorization methods
    providers: t.Optional[t.List[gws.ext.config.authProvider]]  #: authorization providers
    mfa: t.Optional[t.List[gws.ext.config.authMfa]]  #: authorization providers
    session: t.Optional[SessionConfig]  #: session options


class Object(gws.Node, gws.IAuthManager):
    """Authorization manager."""

    sessionOptions: SessionOptions
    guestSession: session.Session
    webMethod: gws.IAuthMethod

    def configure(self):

        so = self.var('session') or SessionOptions()
        so.store = sqlite.SessionStore(so.storePath or SQLITE_STORE_PATH)
        so.lifeTime = so.lifeTime or 1200
        self.sessionOptions = so

        self.providers = self.create_children(gws.ext.object.authProvider, self.var('providers'))
        self.providers.append(self.create_child(gws.ext.object.authProvider, {'type': 'system'}))

        self.guestUser = self.providers[-1].get_user('guest')
        self.systemUser = self.providers[-1].get_user('system')

        self.guestSession = session.Session('guest_session', method=None, user=self.guestUser)

        self.methods = self.create_children(gws.ext.object.authMethod, self.var('methods'))
        if not self.methods:
            # if no methods configured, enable the Web method
            self.methods.append(self.create_child(gws.ext.object.authMethod, {'type': 'web'}))

        self.mfa = self.create_children(gws.ext.object.authMfa, self.var('mfa'))

        for p in self.children:
            p.authMgr = self

    ##

    def activate(self):
        self.root.app.register_web_middleware('auth', self.auth_middleware)

    def auth_middleware(self, req: gws.IWebRequester, nxt) -> gws.IWebResponder:
        self._open_session(req)
        res = nxt()
        self._close_session(req, res)
        return res

    def _open_session(self, req):
        for m in self.methods:
            if m.open_session(req):
                return
        self.session_activate(req, None)

    def _close_session(self, req, res):
        sess = req.session
        if sess and sess.method and sess.method.close_session(req, res):
            return
        self.session_activate(req, None)

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

    def get_provider(self, uid=None):
        return t.cast(gws.IAuthProvider, self.root.get(uid))

    def get_method(self, uid=None, ext_type=None):
        return t.cast(gws.IAuthMethod, self.root.get(uid))

    def get_mfa(self, uid=None, ext_type=None):
        return t.cast(gws.IAuthMfa, self.root.get(uid))

    def serialize_user(self, user):
        return gws.lib.jsonx.to_string([user.provider.uid, user.provider.serialize_user(user)])

    def unserialize_user(self, data):
        provider_uid, ds = gws.lib.jsonx.from_string(data)
        prov = self.get_provider(provider_uid)
        return prov.unserialize_user(ds) if prov else None

    ##

    def session_activate(self, req, sess):
        if not sess:
            sess = self.guestSession
        setattr(req, 'session', sess)
        setattr(req, 'user', sess.user)

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
            data=gws.lib.jsonx.from_string(rec['str_data']),
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
            self.store.update(sess.uid, str_data=gws.lib.jsonx.to_string(sess.data))
        else:
            self.store.touch(sess.uid)

    def session_delete(self, sess: gws.IAuthSession):
        self.store.delete(sess.uid)

    def session_delete_all(self):
        self.store.delete_all()

    def stored_session_records(self) -> t.List[dict]:
        return self.store.get_all()
