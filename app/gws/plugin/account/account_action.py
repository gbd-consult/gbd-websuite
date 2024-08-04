"""User account action."""

from typing import Optional, cast

import gws
import gws.config.util
import gws.base.action
import gws.lib.mime

from . import core, helper

gws.ext.new.action('account')


class Config(gws.base.action.Config):
    """User Account action. (added in 8.1)"""
    pass


class Props(gws.base.action.Props):
    pass


class MfaProps(gws.Data):
    index: int
    title: str
    qrCode: str


class OnboardingStartRequest(gws.Request):
    tc: str


class OnboardingStartResponse(gws.Response):
    tc: str


class OnboardingSavePasswordRequest(gws.Request):
    tc: str
    email: str
    password1: str
    password2: str


class OnboardingSavePasswordResponse(gws.Response):
    tc: str
    ok: bool
    complete: bool
    completionUrl: str
    mfaList: list[MfaProps]


class OnboardingSaveMfaRequest(gws.Request):
    tc: str
    mfaIndex: Optional[int]


class OnboardingSaveMfaResponse(gws.Response):
    complete: bool
    completionUrl: str


class Object(gws.base.action.Object):
    h: helper.Object

    def configure(self):
        self.h = cast(helper.Object, self.root.app.helper('account'))

    @gws.ext.command.api('accountOnboardingStart')
    def onboarding_start(self, req: gws.WebRequester, p: OnboardingStartRequest) -> OnboardingStartResponse:
        account = self.get_account_by_tc(p.tc, core.Category.onboarding, core.Status.new)
        self.h.set_status(account, core.Status.onboarding)
        return OnboardingStartResponse(
            tc=self.h.generate_tc(account, core.Category.onboarding)
        )

    @gws.ext.command.api('accountOnboardingSavePassword')
    def onboarding_save_password(self, req: gws.WebRequester, p: OnboardingSavePasswordRequest) -> OnboardingSavePasswordResponse:
        account = self.get_account_by_tc(p.tc, core.Category.onboarding, core.Status.onboarding)

        p1 = p.password1
        p2 = p.password2

        if account.get('email') != p.email or p1 != p2 or not self.h.validate_password(p1):
            return OnboardingSavePasswordResponse(
                ok=False,
                tc=self.h.generate_tc(account, core.Category.onboarding),
            )

        self.h.set_password(account, p1)

        mfa = self.h.mfa_options(account)
        if mfa:
            mfa_secret = self.h.generate_mfa_secret(account)
            return OnboardingSavePasswordResponse(
                ok=True,
                complete=False,
                mfaList=self.mfa_props(account, mfa_secret),
                tc=self.h.generate_tc(account, core.Category.onboarding),
            )

        self.h.set_status(account, core.Status.active)
        self.h.clear_tc(account)
        return OnboardingSavePasswordResponse(
            ok=True,
            complete=True,
            completionUrl=self.h.onboardingCompletionUrl,
        )

    @gws.ext.command.api('accountOnboardingSaveMfa')
    def onboarding_save_mfa(self, req: gws.WebRequester, p: OnboardingSaveMfaRequest) -> OnboardingSaveMfaResponse:
        account = self.get_account_by_tc(p.tc, core.Category.onboarding, core.Status.onboarding)

        self.h.set_mfa(account, p.mfaIndex)
        self.h.set_status(account, core.Status.active)
        self.h.clear_tc(account)

        return OnboardingSaveMfaResponse(
            complete=True,
            completionUrl=self.h.onboardingCompletionUrl,
        )

    ##

    def get_account_by_tc(self, tc, category, expected_status):
        try:
            account = self.h.get_account_by_tc(tc, category, expected_status)
        except helper.Error as exc:
            raise gws.ForbiddenError() from exc

        if not account:
            raise gws.ForbiddenError(f'account: {tc=} not found')

        return account

    def mfa_props(self, account: dict, mfa_secret):
        ps = []

        for mo in self.h.mfa_options(account):
            ps.append(MfaProps(
                index=mo.index,
                title=mo.title,
                qrCode=self.h.qr_code_for_mfa(account, mo, mfa_secret)
            ))

        return ps
