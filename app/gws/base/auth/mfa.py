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


class OtpConfig:
    """Options for one-time password generation."""

    start: Optional[int]
    """Start time for TOTP counting, as a Unix timestamp."""
    step: Optional[int]
    """TOTP time step in seconds."""
    length: Optional[int]
    """Number of digits in the code."""
    tolerance: Optional[int]
    """Accepted time steps before and after the current one."""
    algo: Optional[str]
    """Hash algorithm for code generation."""


class Config(gws.Config):
    """Multi-factor authorization configuration."""

    message: str = ''
    """Message shown to the user during the MFA step."""
    lifeTime: Optional[gws.Duration] = '120'
    """Time allowed to complete the MFA step."""
    maxVerifyAttempts: int = 3
    """Code entries allowed before the MFA step fails."""
    maxRestarts: int = 0
    """How often a new code can be requested."""
    otp: Optional[OtpConfig]
    """OTP generation options."""


class Object(gws.AuthMultiFactorAdapter):
    otpOptions: gws.lib.otp.Options

    def configure(self):
        self.message = self.cfg('message', default='')
        self.lifeTime = self.cfg('lifeTime', default=120)
        self.maxVerifyAttempts = self.cfg('maxVerifyAttempts', default=3)
        self.maxRestarts = self.cfg('maxRestarts', default=0)
        self.otpOptions = gws.u.merge(gws.lib.otp.DEFAULTS, self.cfg('otp'))

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

    def verify_attempt(self, mfa, payload_valid: bool):
        mfa.verifyCount += 1

        if not self.check_state(mfa):
            return mfa

        if payload_valid:
            mfa.state = gws.AuthMultiFactorState.ok
            return mfa

        if mfa.verifyCount >= self.maxVerifyAttempts:
            mfa.state = gws.AuthMultiFactorState.failed
            return mfa

        mfa.state = gws.AuthMultiFactorState.retry
        return mfa

    def generate_totp(self, mfa: gws.AuthMultiFactorTransaction) -> str:
        ts = self.current_timestamp()
        totp = gws.lib.otp.new_totp(mfa.secret, ts, self.otpOptions)
        mfa.generateTime = ts
        gws.log.debug(f'generate_totp {ts=} {totp=} {mfa.generateTime=}')
        return totp

    def check_totp(self, mfa: gws.AuthMultiFactorTransaction, input: str) -> bool:
        return gws.lib.otp.check_totp(
            str(input or ''),
            mfa.secret,
            self.current_timestamp(),
            self.otpOptions,
        )

    def current_timestamp(self):
        return gws.u.stime()
