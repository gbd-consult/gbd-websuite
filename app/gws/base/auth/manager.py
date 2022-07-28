"""Authorization and session manager."""

import gws
import gws.config
import gws.lib.date
import gws.lib.json2
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

    session_life_time: int
    store: sqlite.SessionStore

    def configure(self):

        self.session_life_time = self.var('sessionLifeTime')

        if self.var('sessionStore') == 'sqlite':
            self.store = sqlite.SessionStore(self.var('sessionStorePath', default=SQLITE_STORE_PATH))
        else:
            # @TODO other store types
            raise gws.ConfigurationError('invalid session store type')

        # always create the System provider
        p = self.var('providers', default=[]) + [
            gws.config.parse(self.root.specs, {'type': 'system'}, gws.ext.config.authProvider)
        ]
        self.providers = self.create_children('gws.ext.auth.provider', p)

        self.guest_user = self.providers[-1].get_user('guest')
        self.system_user = self.providers[-1].get_user('system')

        # if no methods configured, enable the Web method
        web_method = gws.config.parse(self.root.specs, {'type': 'web'}, gws.ext.config.authMethod)
        p = self.var('methods', default=[web_method])
        self.methods = self.create_children('gws.ext.auth.method', p)

    @property
    def guest_session(self):
        return self.new_session('guest_session', method=None, user=self.guest_user)

    def open_session(self, req):
        for m in self.methods:
            sess = m.open_session(self, req)
            if sess:
                return sess
        return self.guest_session

    def close_session(self, sess, req, res):
        if sess and sess.method:
            return sess.method.close_session(self, sess, req, res)
        return self.guest_session

    def authenticate(self, method, credentials):
        for prov in self.providers:
            if prov.allowed_methods and method.ext_type not in prov.allowed_methods:
                continue
            gws.log.debug(f'trying provider {prov.uid!r}')
            user = prov.authenticate(method, credentials)
            if user:
                return user

    def get_user(self, user_uid):
        provider_uid, local_uid = user_api.parse_uid(user_uid)
        prov = self.get_provider(provider_uid)
        if prov:
            return prov.get_user(local_uid)

    def get_provider(self, uid=None, ext_type=None):
        for obj in self.providers:
            if (uid and obj.uid == uid) or (ext_type and obj.ext_type == ext_type):
                return obj

    def get_method(self, uid=None, ext_type=None):
        for obj in self.methods:
            if (uid and obj.uid == uid) or (ext_type and obj.ext_type == ext_type):
                return obj

    def serialize_user(self, user):
        return gws.lib.json2.to_string([user.provider.uid, user.provider.serialize_user(user)])

    def unserialize_user(self, ser):
        provider_uid, str_user = gws.lib.json2.from_string(ser)
        prov = self.get_provider(provider_uid)
        return prov.unserialize_user(str_user) if prov else None

    # private API for auth methods

    def new_session(self, typ, user, method=None, uid=None, data=None) -> gws.IAuthSession:
        return session.Session(typ, user, method, uid, data)

    def find_stored_session(self, uid: str) -> t.Optional[gws.IAuthSession]:
        rec = self.store.find(uid)
        if not rec:
            return None

        age = gws.lib.date.timestamp() - rec['updated']
        if age > self.session_life_time:
            gws.log.debug(f'sess uid={uid!r} EXPIRED age={age!r}')
            self.store.delete(uid)
            return None

        user = self.unserialize_user(rec['str_user'])
        if not user:
            gws.log.error(f'FAILED to unserialize user from sess={uid!r}')
            self.store.delete(uid)
            return None

        return self.new_session(
            rec['typ'],
            uid=rec['uid'],
            method=self.get_method(rec['method_uid']),
            user=user,
            data=gws.lib.json2.from_string(rec['str_data'])
        )

    def new_stored_session(self, typ: str, method: gws.IAuthMethod, user: gws.IUser) -> gws.IAuthSession:
        self.store.cleanup(self.session_life_time)

        uid = self.store.create(
            typ=typ,
            method_uid=method.uid,
            provider_uid=user.provider.uid,
            user_uid=user.uid,
            str_user=self.serialize_user(user))

        sess = self.find_stored_session(uid)
        if not sess:
            raise gws.Error('failed to create a new session')

        return sess

    def save_stored_session(self, sess: gws.IAuthSession):
        if sess.changed:
            self.store.update(sess.uid, str_data=gws.lib.json2.to_string(sess.data))
        else:
            self.store.touch(sess.uid)

    def destroy_stored_session(self, sess: gws.IAuthSession):
        self.store.delete(sess.uid)

    def delete_stored_sessions(self):
        self.store.delete_all()

    def stored_session_records(self) -> t.List[dict]:
        return self.store.get_all()
