"""Authorization and session manager."""

import gws
import gws.common.auth.error
import gws.common.auth.session
import gws.common.auth.stores.sqlite
import gws.common.auth.user
import gws.tools.date
import gws.tools.json2

import gws.types as t

from .error import Error


class Config(t.Config):
    """Authentication and authorization options"""

    methods: t.Optional[t.List[t.ext.auth.method.Config]]  #: authorization methods
    providers: t.List[t.ext.auth.provider.Config]  #: authorization providers
    mfa: t.Optional[t.List[t.ext.auth.mfa.Config]]  #: multifactor authorization plugins
    sessionLifeTime: t.Duration = 1200  #: session life time
    sessionStorage: str = 'sqlite'  #: session storage engine


#:export IAuthManager
class Object(gws.Object, t.IAuthManager):
    """Authorization manager."""

    def configure(self):
        super().configure()

        self.session_life_time = self.var('sessionLifeTime')

        # if self.var('sessionStorage') == 'sqlite':
        # @TODO other store types
        self.store = gws.common.auth.stores.sqlite.SessionStore()
        self.store.init()

        p = self.var('providers', default=[])
        self.providers: t.List[t.IAuthProvider] = [
            t.cast(t.IAuthProvider, self.create_child('gws.ext.auth.provider', c))
            for c in p]

        sys: t.IAuthProvider = t.cast(t.IAuthProvider, self.create_child('gws.ext.auth.provider', t.Config(type='system')))
        self.providers.append(sys)
        self.guest_user: t.IUser = sys.get_user('guest')

        # no methods at all, enable the web method
        p = self.var('methods', default=[t.Config(type='web')])
        self.methods: t.List[t.IAuthMethod] = [
            t.cast(t.IAuthMethod, self.create_child('gws.ext.auth.method', c))
            for c in p]

        p = self.var('mfa', default=[])
        self.mfas: t.List[t.IAuthMfa] = [
            t.cast(t.IAuthMfa, self.create_child('gws.ext.auth.mfa', c))
            for c in p]

        for c in self.children:
            c.auth = self

    @property
    def guest_session(self):
        return self.new_session(type='guest', user=self.guest_user)

    # session manager

    def new_session(self, **kwargs):
        return gws.common.auth.session.Session(**kwargs)

    def open_session(self, req: t.IRequest) -> t.ISession:
        for m in self.methods:
            sess = m.open_session(self, req)
            if sess:
                return sess
        return self.guest_session

    def close_session(self, sess: t.ISession, req: t.IRequest, res: t.IResponse) -> t.ISession:
        if sess and sess.method:
            return sess.method.close_session(self, sess, req, res)
        return self.guest_session

    # stored sessions

    def find_stored_session(self, uid):
        rec = self.store.find(uid)
        if not rec:
            return

        age = gws.tools.date.timestamp() - rec['updated']
        if age > self.session_life_time:
            gws.log.debug(f'sess uid={uid!r} EXPIRED age={age!r}')
            self.store.delete(uid)
            return

        return self.new_session(
            type=rec['session_type'],
            uid=rec['uid'],
            method=self.get_method(rec['method_type']),
            user=self.unserialize_user(rec['str_user']),
            data=gws.tools.json2.from_string(rec['str_data'])
        )

    def create_stored_session(self, type: str, method: t.IAuthMethod, user: t.IUser) -> t.ISession:
        self.store.cleanup(self.session_life_time)

        uid = self.store.create(
            session_type=type,
            method_type=method.type,
            provider_uid=user.provider.uid,
            user_uid=user.uid,
            str_user=self.serialize_user(user))

        return self.find_stored_session(uid)

    def save_stored_session(self, sess: t.ISession):
        if sess.changed:
            self.store.update(sess.uid, str_data=gws.tools.json2.to_string(sess.data))
        else:
            self.store.touch(sess.uid)

    def destroy_stored_session(self, sess: t.ISession):
        gws.log.debug(f'destroy_stored_session uid={sess.uid!r}')
        self.store.delete(sess.uid)

    def delete_stored_sessions(self):
        self.store.delete_all()

    def stored_session_records(self) -> t.List[dict]:
        return self.store.get_all()

    #

    def authenticate(self, method: t.IAuthMethod, login, password, **kw) -> t.Optional[t.IUser]:
        for prov in self.providers:
            if prov.allowed_methods and method.type not in prov.allowed_methods:
                continue
            gws.log.debug(f'trying provider {prov.uid!r} for login {login!r}')
            user = prov.authenticate(method, login, password, **kw)
            if user:
                return user

    def get_user(self, user_fid: str) -> t.Optional[t.IUser]:
        provider_uid, user_uid = gws.common.auth.user.parse_fid(user_fid)
        prov = self.get_provider(provider_uid)
        if prov:
            return prov.get_user(user_uid)

    def get_role(self, name: str) -> t.IRole:
        return gws.common.auth.user.Role(name)

    def get_provider(self, uid: str) -> t.Optional[t.IAuthProvider]:
        for prov in self.providers:
            if prov.uid == uid:
                return prov

    def get_method(self, type: str) -> t.Optional[t.IAuthMethod]:
        for m in self.methods:
            if m.type == type:
                return m

    def get_mfa(self, uid: str) -> t.Optional[t.IAuthMfa]:
        for m in self.mfas:
            if m.uid == uid:
                return m

    def serialize_user(self, user: t.IUser) -> str:
        return gws.tools.json2.to_string(user.provider.user_to_dict(user))

    def unserialize_user(self, s: str) -> t.IUser:
        d = gws.tools.json2.from_string(s)
        prov = self.get_provider(d['provider_uid'])
        return prov.user_from_dict(d)
