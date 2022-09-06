"""TOTP two factor auth method."""

import gws
import gws.common.auth.mfa
import gws.web.error

import gws.types as t

from gws.common.auth.mfa import Error


class Config(gws.common.auth.mfa.Config):
    pass


class Object(gws.common.auth.mfa.Object):
    def start(self, user):
        mf = super().start(user)
        if not mf.secret:
            raise Error(f'secret required, user {user.uid!r}')
        return mf

    def verify_attempt(self, user, mf, data):
        return self.check_totp(mf, data.token)
