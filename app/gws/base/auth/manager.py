"""Authorization manager."""

from typing import Optional, cast

import gws
import gws.config
import gws.lib.jsonx

from . import session, system_provider, throttle


class Config(gws.Config):
    """Authentication methods, providers, sessions and login throttling."""

    methods: Optional[list[gws.ext.config.authMethod]]
    """Login methods available to users."""
    providers: Optional[list[gws.ext.config.authProvider]]
    """User sources that verify credentials."""
    mfa: Optional[list[gws.ext.config.authMultiFactorAdapter]]
    """Multi-factor authentication adapters."""
    session: Optional[gws.ext.config.authSessionManager]
    """Session storage and life time."""
    throttle: Optional[throttle.Config]
    """Blocking of repeated failed login attempts, disabled if not set."""


_DEFAULT_SESSION_TYPE = 'sqlite'


class Object(gws.AuthManager):
    """Authorization manager."""

    def configure(self):
        self.sessionMgr = self.create_child(gws.ext.object.authSessionManager, self.cfg('session'), type=_DEFAULT_SESSION_TYPE)

        self.throttle = None
        p = self.cfg('throttle')
        if p:
            self.throttle = cast(throttle.Object, self.create_child(throttle.Object, p))

        self.providers = self.create_children(gws.ext.object.authProvider, self.cfg('providers'))

        sys_provider = self.create_child(system_provider.Object)
        self.add_provider(sys_provider)

        self.guestUser = sys_provider.get_user('guest')
        self.systemUser = sys_provider.get_user('system')

        self.methods = self.create_children(gws.ext.object.authMethod, self.cfg('methods'))
        if not self.methods:
            # if no methods configured, enable the Web method
            self.add_method(self.create_child(gws.ext.object.authMethod, type='web'))

        self.mfAdapters = self.create_children(gws.ext.object.authMultiFactorAdapter, self.cfg('mfa'))

        self.guestSession = session.Object(uid='guest_session', method=None, user=self.guestUser)

        self.root.app.middlewareMgr.register(self, 'auth', depends_on=['db'])

    ##

    def add_provider(self, provider):
        self.providers.append(provider)

    def add_method(self, method):
        self.methods.append(method)

    def add_multi_factor_adapter(self, adapter):
        self.mfAdapters.append(adapter)

    ##

    def enter_middleware(self, req: gws.WebRequester):
        sess = self._try_open_session(req)
        if sess:
            req.set_session(sess)
            gws.log.debug(f'session: user={req.session.user.uid!r} roles={req.session.user.roles}')
        else:
            gws.log.debug('session: guest')
            req.set_session(self.guestSession)

    def _try_open_session(self, req):
        """Return the first session opened by a usable method, or ``None``.

        A method that requires a secure context is skipped on insecure requests,
        unless the client address is in its ``allowInsecureFrom``. A session whose
        user no longer exists or whose method does not match the opening method is
        deleted, and ``None`` is returned, so the request continues as guest.
        """
        for meth in self.methods:
            if not self.can_use_method(req, meth):
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

    def exit_middleware(self, req: gws.WebRequester, res: gws.WebResponder):
        sess = req.session
        if sess.method:
            sess.method.close_session(req, res)
        req.set_session(self.guestSession)

    def can_use_method(self, req, meth):
        if not meth.secure or req.isSecure:
            return True
        if not meth.allowInsecureFrom:
            return False
        if req.ip not in meth.allowInsecureFrom:
            return False
        gws.log.warning(f'open_session: {meth=}: insecure_context allowed from {req.ip=}')
        return True

    ##

    def create_transient_session(self, method, user, data=None):
        return session.Object(
            uid=gws.u.random_string(64),
            method=method,
            user=user,
            data=data,
            is_changed=False,
            is_transient=True,
        )

    ##

    def authenticate(self, method, credentials, req):
        if not self.throttle:
            return self._authenticate2(method, credentials)

        sec = self.throttle.blocked_for(req, method, credentials)
        if sec > 0:
            gws.log.warning(f'authenticate: {method=}: throttled for {sec}s')
            exc = gws.TooManyRequestsError('too many authentication attempts')
            exc.retryAfter = sec
            raise exc

        try:
            user = self._authenticate2(method, credentials)
        except gws.ForbiddenError:
            self.throttle.register(False, req, method, credentials)
            raise

        self.throttle.register(user is not None, req, method, credentials)
        return user

    def _authenticate2(self, method, credentials):
        """Try each provider that allows the method, return the first user found."""
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

    def get_method(self, uid=None):
        for obj in self.methods:
            if obj.uid == uid:
                return obj

    def get_multi_factor_adapter(self, uid=None):
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
