"""MFA method"""

import pyotp

import gws
import gws.lib.date
import gws.base.auth.mfa

gws.ext.new.authMfa('email')


class Config(gws.base.auth.mfa.Config):
    """Web-based authorization options"""

    cookieName: str = 'auth'
    """name for the cookie"""
    cookiePath: str = '/'
    """cookie path"""


class Object(gws.base.auth.mfa.Object):

    def start(self, user):
        pm = user.pendingMfa
        pm.timeStarted = gws.lib.date.timestamp()
        pm.attemptCount = 0
        pm.restartCount = 0
        pm.secret = pm.secret or pyotp.random_base32()
        print('SEC', pm.secret)
        self.generate_and_send(user)

    def is_valid(self, user):
        pm = user.pendingMfa
        td = pm.timeStarted - gws.lib.date.timestamp()
        return True

    def verify(self, user, request):
        pm = user.pendingMfa
        pm.attemptCount += 1
        obj = pyotp.TOTP(pm.secret)
        return obj.verify(gws.u.get(request, 'otp'))

    def restart(self, user):
        pm = user.pendingMfa
        pm.restartCount += 1
        self.generate_and_send(user)

    def generate_and_send(self, user):
        pm = user.pendingMfa
        obj = pyotp.TOTP(pm.secret)
        otp = obj.now()

        print('MFA=', obj.now())
