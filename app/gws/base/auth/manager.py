"""Authorization and session manager."""

import gws
import gws.config
import gws.lib.date
import gws.lib.jsonx
import gws.types as t

from . import session, system_provider


class Config(gws.Config):
    """Authentication and authorization options"""

    methods: t.Optional[list[gws.ext.config.authMethod]]
    """authorization methods"""
    providers: t.Optional[list[gws.ext.config.authProvider]]
    """authorization providers"""
    mfa: t.Optional[list[gws.ext.config.authMfa]]
    """authorization providers"""
    session: t.Optional[gws.ext.config.authSessionManager]
    """session options"""


_DEFAULT_SESSION_TYPE = 'sqlite'


class Object(gws.Node, gws.IAuthManager):
    """Authorization manager."""

    def configure(self):
        self.sessionMgr = self.create_child(gws.ext.object.authSessionManager, self.cfg('session'), type=_DEFAULT_SESSION_TYPE, _defaultManager=self)

        self.providers = self.create_children(gws.ext.object.authProvider, self.cfg('providers'), _defaultManager=self)

        sys_provider = self.create_child(system_provider.Object, _defaultManager=self)
        self.providers.append(sys_provider)

        self.guestUser = sys_provider.get_user('guest')
        self.systemUser = sys_provider.get_user('system')

        self.methods = self.create_children(gws.ext.object.authMethod, self.cfg('methods'), _defaultManager=self)
        if not self.methods:
            # if no methods configured, enable the Web method
            self.methods.append(self.create_child(gws.ext.object.authMethod, type='web', _defaultManager=self))

        self.mfa = self.create_children(gws.ext.object.authMfa, self.cfg('mfa'), _defaultManager=self)

        self.guestSession = session.Object(uid='guest_session', method=None, user=self.guestUser)

        self.root.app.register_middleware('auth', self, depends_on=['db'])

    ##

    def enter_middleware(self, req):
        req.set_session(self.guestSession)
        for meth in self.methods:
            gws.log.debug(f'trying method {meth}')
            sess = meth.open_session(req)
            if sess:
                gws.log.debug(f'ok method {meth}')
                req.set_session(sess)
                break
        gws.log.debug(f'session opened: user={req.session.user.uid!r} roles={req.session.user.roles}')

    def exit_middleware(self, req, res):
        sess = req.session
        if sess.method:
            sess.method.close_session(req, res)
        req.set_session(self.guestSession)

    ##

    def authenticate(self, method, credentials):
        for prov in self.providers:
            if prov.allowedMethods and method.extType not in prov.allowedMethods:
                continue
            gws.log.debug(f'trying provider {prov!r}')
            user = prov.authenticate(method, credentials)
            if user:
                gws.log.debug(f'ok provider {prov!r}')
                return user

    ##

    def get_user(self, user_uid):
        provider_uid, local_uid = gws.split_uid(user_uid)
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
