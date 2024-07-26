"""TOTP multi-factor adapter.

This adapter accepts TOTPs from authenticator apps.
The user is required to have an ``mfaSecret`` string attribute.


"""

import gws
import gws.base.auth
import gws.lib.net
import gws.lib.otp

gws.ext.new.authMultiFactorAdapter('totp')


class Config(gws.base.auth.mfa.Config):
    pass


class Object(gws.base.auth.mfa.Object):

    def start(self, user):
        if not user.mfaSecret:
            gws.log.warning(f'totp: cannot start, {user.uid=}: no secret')
            return

        mfa = super().start(user)
        mfa.secret = user.mfaSecret

        return mfa

    def verify(self, mfa, payload):
        ok = self.check_totp(mfa, payload['code'])
        return self.verify_attempt(mfa, ok)
