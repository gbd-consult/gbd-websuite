"""Authorization and session manager."""

from typing import Optional, cast

import gws
import gws.config
import gws.lib.jsonx

from . import session, system_provider


class Config(gws.Config):
    """Authentication and authorization options"""

    methods: Optional[list[gws.ext.config.authMethod]]
    """authorization methods"""
    providers: Optional[list[gws.ext.config.authProvider]]
    """authorization providers"""
    mfa: Optional[list[gws.ext.config.authMultiFactorAdapter]]
    """authorization providers"""
    session: Optional[gws.ext.config.authSessionManager]
    """session options"""


_DEFAULT_SESSION_TYPE = 'sqlite'


class Object(gws.AuthManager):
    """Authorization manager."""

    def configure(self):
        self.sessionMgr = self.create_child(gws.ext.object.authSessionManager, self.cfg('session'), type=_DEFAULT_SESSION_TYPE)

        self.providers = self.create_children(gws.ext.object.authProvider, self.cfg('providers'))

        sys_provider = self.create_child(system_provider.Object)
        self.providers.append(sys_provider)

        self.guestUser = sys_provider.get_user('guest')
        self.systemUser = sys_provider.get_user('system')

        self.methods = self.create_children(gws.ext.object.authMethod, self.cfg('methods'))
        if not self.methods:
            # if no methods configured, enable the Web method
            self.methods.append(self.create_child(gws.ext.object.authMethod, type='web'))

        self.mfAdapters = self.create_children(gws.ext.object.authMultiFactorAdapter, self.cfg('mfa'))

        self.guestSession = session.Object(uid='guest_session', method=None, user=self.guestUser)

        self.root.app.middlewareMgr.register(self, 'auth', depends_on=['db'])

    ##

    def enter_middleware(self, req):
        sess = self._try_open_session(req) or self.guestSession
        req.set_session(sess)
        gws.log.debug(f'session opened: user={req.session.user.uid!r} roles={req.session.user.roles}')

    def _try_open_session(self, req):
        for meth in self.methods:
            if meth.secure and not req.isSecure:
                gws.log.warning(f'open_session: {meth=}: insecure_context, ignore')
                continue

            sess = meth.open_session(req)
            if not sess:
                continue

            if not sess.user:
                gws.log.warning(f'open_session: {meth=}: {sess.uid=} user not found')
                self.sessionMgr.delete(sess)
                return

            if not sess.method or sess.method.uid != meth.uid:
                gws.log.warning(f'open_session: {meth=}: {sess.uid=} wrong method {sess.method=}')
                self.sessionMgr.delete(sess)
                return

            gws.log.debug(f'open_session: {meth=}: ok')
            return sess

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
        provider_uid, local_uid = gws.u.split_uid(user_uid)
        prov = self.get_provider(provider_uid)
        return prov.get_user(local_uid) if prov else None

    def get_provider(self, uid):
        for obj in self.providers:
            if obj.uid == uid:
                return obj

    def get_method(self, uid=None, ext_type=None):
        for obj in self.methods:
            if obj.uid == uid:
                return obj

    def get_multi_factor_adapter(self, uid=None, ext_type=None):
        for obj in self.mfAdapters:
            if obj.uid == uid:
                return obj

    def serialize_user(self, user):
        return gws.lib.jsonx.to_string([user.authProvider.uid, user.authProvider.serialize_user(user)])

    def unserialize_user(self, data):
        provider_uid, ds = gws.lib.jsonx.from_string(data)
        prov = self.get_provider(provider_uid)
        return prov.unserialize_user(ds) if prov else None

    def is_public_object(self, obj, *context):
        return self.guestUser.can_read(obj, *context)
