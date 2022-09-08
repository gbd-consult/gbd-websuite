"""MFA method"""

import random
import time

import gws
import gws.tools.date
import gws.common.template
import gws.tools.otp

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


class Config(t.WithType):
    """MFA method config."""

    lifeTime: t.Optional[t.Duration] = 1200  #: how long to wait for the MFA to complete
    maxRestarts: int = 0  #: max code regeneration attempts
    maxVerifyAttempts: int = 3  #: max verify attempts
    templates: t.Optional[t.List[t.ext.template.Config]]  #: client and email templates
    totpStep: t.Duration = 30  #: default time step as per rfc6238
    totpStart: int = 0  #: default initial time as per rfc6238
    totpLength: int = 6  #: token length


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
            generatedTime=gws.tools.date.timestamp(),
        )

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
        s = self.get_totp(mf, ts=int(time.time()))
        mf.generatedTime = gws.tools.date.timestamp()
        gws.log.debug(f'generate_totp totp={s} t={mf.generatedTime}')
        return s

    def check_totp(self, mf: t.AuthMfaData, token: str) -> bool:
        token = str(token or '')
        if len(token) != mf.totpLength:
            return False

        ts = int(time.time())

        for window in [0, -1, +1]:
            s = self.get_totp(mf, ts + mf.totpStep * window)
            gws.log.debug(f'check_totp tok={token} totp={s} w={window} step={mf.totpStep} gt={mf.generatedTime} t={gws.tools.date.timestamp()}')
            if token == s:
                return True

        return False

    def get_totp(self, mf, ts):
        return gws.tools.otp.totp(
            mf.secret,
            timestamp=ts,
            start=mf.totpStart,
            step=mf.totpStep,
            length=mf.totpLength,
        )
