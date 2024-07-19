"""Generic multi-factor authentication adapter.

Multi-factor authentication (handled in ``gws.plugin.auth_method.web.core`)
is used for ``User`` object that provide the attribute ``mfaUid``,
which is supposed to be an ID of a configured MFA Adapter.

Specific MFA Adapters can require other attributes.

Multi-factor authentication starts by creating a `gws.AuthMultiFactorTransaction` object,
kept in a session until it is verified or expires.

Some Adapters can be restarted (e.g. by resending a verification email).
"""

from typing import Optional

import gws
import gws.lib.otp
import gws.lib.net


class OtpOptions(gws.Data):
    start: int = 0
    step: int = 30
    length: int = 6
    tolerance: int = 1
    algo: str = 'sha1'


class Config(gws.Config):
    """Multi-factor authorization configuration."""

    message: str = ''
    """Message to display in the client."""
    lifeTime: Optional[gws.Duration] = 120
    """How long to wait for the MFA to complete."""
    maxVerifyAttempts: int = 3
    """Max verify attempts."""
    maxRestarts: int = 0
    """Max code regeneration attempts."""
    otp: Optional[OtpOptions]
    """OTP generation options"""


class Object(gws.AuthMultiFactorAdapter):
    OTP_DEFAULTS = OtpOptions(
        start=0,
        step=30,
        length=6,
        tolerance=1,
        algo='sha1',
    )

    otp: OtpOptions

    def configure(self):
        self.message = self.cfg('message', default='')
        self.lifeTime = self.cfg('lifeTime', default=120)
        self.maxVerifyAttempts = self.cfg('maxVerifyAttempts', default=3)
        self.maxRestarts = self.cfg('maxRestarts', default=0)

        p = gws.u.to_dict(self.cfg('otp'))
        self.otp = OtpOptions(
            start=p.get('start', self.OTP_DEFAULTS.start),
            step=p.get('step', self.OTP_DEFAULTS.step),
            length=p.get('length', self.OTP_DEFAULTS.length),
            tolerance=p.get('tolerance', self.OTP_DEFAULTS.tolerance),
            algo=p.get('algo', self.OTP_DEFAULTS.algo),
        )

    def start(self, user):
        return gws.AuthMultiFactorTransaction(
            state=gws.AuthMultiFactorState.open,
            restartCount=0,
            verifyCount=0,
            secret='',
            startTime=self.current_timestamp(),
            generateTime=0,
            message=self.message,
            adapter=self,
            user=user,
        )

    def check_state(self, mfa):
        ts = self.current_timestamp()
        if ts - mfa.startTime >= self.lifeTime:
            mfa.state = gws.AuthMultiFactorState.failed
            return False
        if mfa.verifyCount > self.maxVerifyAttempts:
            mfa.state = gws.AuthMultiFactorState.failed
            return False
        if mfa.state == gws.AuthMultiFactorState.failed:
            return False
        return True

    def verify(self, mfa, payload):
        mfa.verifyCount += 1

        if not self.check_state(mfa):
            return mfa

        ok = self.check_payload(mfa, payload)
        if ok:
            mfa.state = gws.AuthMultiFactorState.ok
            return mfa

        if mfa.verifyCount >= self.maxVerifyAttempts:
            mfa.state = gws.AuthMultiFactorState.failed
            return mfa

        mfa.state = gws.AuthMultiFactorState.retry
        return mfa

    def check_restart(self, mfa):
        return mfa.restartCount < self.maxRestarts

    def restart(self, mfa):
        rc = mfa.restartCount + 1
        if rc > self.maxRestarts:
            return

        mfa = self.start(mfa.user)
        if not mfa:
            return

        mfa.restartCount = rc
        return mfa

    ##

    def generic_key_uri(self, method: str, secret: str, issuer_name: str, account_name: str, counter: Optional[int] = None) -> str:
        """Create a key uri for auth apps.

        Reference:
            https://github.com/google/google-authenticator/wiki/Key-Uri-Format
        """

        params = {
            'secret': gws.lib.otp.base32_encode(secret),
            'issuer': issuer_name,
        }
        if self.otp.algo != self.OTP_DEFAULTS.algo:
            params['algorithm'] = self.otp.algo
        if self.otp.length != self.OTP_DEFAULTS.length:
            params['digits'] = self.otp.length
        if self.otp.step != self.OTP_DEFAULTS.step:
            params['period'] = self.otp.step
        if counter is not None:
            params['counter'] = counter

        return 'otpauth://{}/{}:{}?{}'.format(
            method,
            gws.lib.net.quote_param(issuer_name),
            gws.lib.net.quote_param(account_name),
            gws.lib.net.make_qs(params)
        )

    def generate_totp(self, mfa: gws.AuthMultiFactorTransaction) -> str:
        ts = self.current_timestamp()
        totp = self.new_totp(mfa, ts)
        mfa.generateTime = ts
        gws.log.debug(f'generate_totp {ts=} {totp=} {mfa.generateTime=}')
        return totp

    def check_totp(self, mfa: gws.AuthMultiFactorTransaction, code: str) -> bool:
        ts = self.current_timestamp()

        code = str(code or '')
        if len(code) != self.otp.length:
            return False

        for window in range(-self.otp.tolerance, self.otp.tolerance + 1):
            totp = self.new_totp(mfa, ts + self.otp.step * window)
            gws.log.debug(f'check_totp {ts=} {totp=} {code=} {window=} {mfa.generateTime=}')
            if code == totp:
                return True

        return False

    def new_totp(self, mfa: gws.AuthMultiFactorTransaction, timestamp: int):
        return gws.lib.otp.new_totp(
            mfa.secret,
            timestamp=timestamp,
            start=self.otp.start,
            step=self.otp.step,
            length=self.otp.length,
            algo=self.otp.algo,
        )

    def current_timestamp(self):
        return gws.u.stime()
