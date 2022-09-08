"""MFA method"""

import random
import time

import gws
import gws.tools.date
import gws.common.template
import gws.tools.vendor.onetimepass

import gws.types as t


class Error(gws.Error):
    pass


#:export
class AuthMfaData(t.Data):
    uid: str
    secret: str
    verifyCount: int
    restartCount: int
    startTime: int
    totpStart: int
    totpStep: int
    clientOptions: dict


class Config(t.WithType):
    """MFA method config."""

    lifeTime: t.Optional[t.Duration] = 1200  #: how long to wait for the MFA to complete
    maxRestarts: int = 0  #: max code regeneration attempts
    maxVerifyAttempts: int = 3  #: max verify attempts
    templates: t.Optional[t.List[t.ext.template.Config]]  #: message templates
    totpStep: t.Duration = 30  #: default time step as per rfc6238
    totpStart: int = 0  #: default initial time as per rfc6238
    totpLength: int = 6  #: default initial time as per rfc6238


#:export IAuthMfa
class Object(gws.Object, t.IAuthMfa):
    auth: t.IAuthManager

    def configure(self):
        super().configure()

        self.type: str = self.var('type')
        self.uid: str = self.var('uid')

        self.lifeTime = self.var('lifeTime')
        self.maxVerifyAttempts = self.var('maxVerifyAttempts')
        self.maxRestarts = self.var('maxRestarts')

        self.totpStep = self.var('totpStep')
        self.totpStart = self.var('totpStart')
        self.totpLength = self.var('totpLength')

        self.templates: t.List[t.ITemplate] = gws.common.template.bundle(self, self.var('templates'))

    def start(self, user: t.IUser) -> t.AuthMfaData:
        mf = t.AuthMfaData(
            uid=user.attribute('mfauid'),
            secret=user.attribute('mfasecret'),
            totpStart=gws.as_int(user.attribute('mfatotpstart')) or self.totpStart,
            totpStep=gws.as_int(user.attribute('mfatotpstep')) or self.totpStep,
            totpLength=self.totpLength,
            restartCount=0,
            verifyCount=0,
            startTime=gws.tools.date.timestamp(),
            clientOptions={}
        )

        tpl = gws.common.template.find(self.templates, subject='client.message')
        mf.clientOptions['message'] = tpl.render({'user': user}).content

        # NB do not store the secret
        user.attributes.pop('mfasecret', None)

        gws.log.debug(f'MFA: started, uid={self.uid!r} user={user.uid!r}')

        return mf

    def restart(self, user: t.IUser, mf: t.AuthMfaData):
        raise Error('restart not supported')

    def verify(self, user: t.IUser, mf: t.AuthMfaData, data: t.Data) -> bool:
        if mf.verifyCount >= self.maxVerifyAttempts:
            raise Error('too many verify attempts')

        ok = self.verify_attempt(user, mf, data)

        mf.verifyCount += 1

        if not ok and mf.verifyCount >= self.maxVerifyAttempts:
            raise Error('too many verify attempts')

        return ok

    def verify_attempt(self, user: t.IUser, mf: t.AuthMfaData, data: t.Data) -> bool:
        return False

    def is_valid(self, user: t.IUser, mf: t.AuthMfaData) -> bool:
        return gws.tools.date.timestamp() - mf.startTime < self.lifeTime

    def generate_totp(self, mf: t.AuthMfaData) -> str:
        int_totp = self.get_totp(mf, int(time.time() - mf.totpStart))
        mf.generatedTime = gws.tools.date.timestamp()
        gws.log.debug(f'generate_totp totp={int_totp} t={mf.generatedTime}')
        return '{:0{}d}'.format(int_totp, mf.totpLength)

    def check_totp(self, mf: t.AuthMfaData, token: str) -> bool:
        try:
            int_token = int(token)
        except ValueError:
            return False

        clock = int(time.time() - mf.totpStart)
        for window in [0, -1, +1]:
            int_totp = self.get_totp(mf, clock + mf.totpStep * window)
            gws.log.debug(f'check_totp tok={int_token} totp={int_totp} w={window} step={mf.totpStep} gt={mf.generatedTime} t={gws.tools.date.timestamp()}')
            if int_token == int_totp:
                return True

        return False

    def get_random_secret(self, length: int = 32) -> str:
        _b32alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'  # from base64.py
        r = random.SystemRandom()
        return ''.join(r.choice(_b32alphabet) for _ in range(length))

    def get_totp(self, mf, clock):
        return gws.tools.vendor.onetimepass.get_totp(
            mf.secret,
            clock=clock,
            interval_length=mf.totpStep,
            token_length=mf.totpLength,
        )
