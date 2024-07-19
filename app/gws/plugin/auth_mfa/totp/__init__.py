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

    def check_payload(self, mfa, payload):
        return self.check_totp(mfa, payload['code'])

    def key_uri(self, user: gws.User, issuer_name: str, account_name: str = '') -> str:
        """Create a key uri for auth apps."""

        if not user.mfaSecret:
            raise gws.Error(f'{user.uid=}: no secret')

        return self.generic_key_uri(
            method='totp',
            secret=user.mfaSecret,
            issuer_name=issuer_name,
            account_name=account_name or user.loginName,
        )
