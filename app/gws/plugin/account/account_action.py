"""Action for the account onboarding procedure."""

from typing import Optional, cast

import gws
import gws.config.util
import gws.base.action
import gws.lib.mime

from . import core, helper


@gws.ext.config.action('account')
class Config(gws.base.action.Config):
    """Account management for end users, including onboarding."""
    pass


@gws.ext.props.action('account')
class Props(gws.base.action.Props):
    pass


class MfaProps(gws.Data):
    """MFA method offered to the user during onboarding."""

    index: int
    """Index of the method in the helper's ``mfa`` list, starting with 1."""
    title: str
    """Title of the method."""
    qrCode: str
    """QR code of the key URI as a data URL, empty if the method has no key URI."""


class OnboardingStartRequest(gws.Request):
    """Request to start the onboarding."""

    tc: str
    """Temporary code from the onboarding link."""


class OnboardingStartResponse(gws.Response):
    """Response to the onboarding start."""

    tc: str
    """New temporary code for the next step."""


class OnboardingSavePasswordRequest(gws.Request):
    """Request to set the password during onboarding."""

    tc: str
    """Temporary code from the previous step."""
    email: str
    """Email address, must match the account's email."""
    password1: str
    """New password."""
    password2: str
    """New password, repeated."""


class OnboardingSavePasswordResponse(gws.Response):
    """Response to setting the password."""

    tc: str
    """New temporary code for the next step, if the onboarding is not complete."""
    ok: bool
    """``False`` if the email does not match or the password is invalid."""
    complete: bool
    """``True`` if the account is active, ``False`` if an MFA method must be selected."""
    completionUrl: str
    """URL to go to when the onboarding is complete."""
    mfaList: list[MfaProps]
    """MFA methods to choose from."""


class OnboardingSaveMfaRequest(gws.Request):
    """Request to select an MFA method during onboarding."""

    tc: str
    """Temporary code from the previous step."""
    mfaIndex: Optional[int]
    """Index of the selected method."""


class OnboardingSaveMfaResponse(gws.Response):
    """Response to selecting an MFA method."""

    complete: bool
    """``True`` when the onboarding is complete."""
    completionUrl: str
    """URL to go to when the onboarding is complete."""


@gws.ext.object.action('account')
class Object(gws.base.action.Object):
    """Account action, provides the onboarding API for the client.

    The onboarding steps are: start, set the password, select an MFA method (only if MFA methods are configured).
    Each step requires the temporary code returned by the previous step, the first one the code from the onboarding link.
    """

    h: helper.Object
    """The account helper."""

    def configure(self):
        self.h = cast(helper.Object, self.root.app.helper('account'))

    @gws.ext.command.api('accountOnboardingStart')
    def onboarding_start(self, req: gws.WebRequester, p: OnboardingStartRequest) -> OnboardingStartResponse:
        """Start the account onboarding."""

        account = self.get_account_by_tc(p.tc, core.Category.onboarding, core.Status.new)
        self.h.set_status(account, core.Status.onboarding)
        return OnboardingStartResponse(
            tc=self.h.generate_tc(account, core.Category.onboarding)
        )

    @gws.ext.command.api('accountOnboardingSavePassword')
    def onboarding_save_password(self, req: gws.WebRequester, p: OnboardingSavePasswordRequest) -> OnboardingSavePasswordResponse:
        """Set the password of the account."""

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
        """Save the selected MFA method and activate the account."""

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
        """Find an account by a temporary code.

        The code is invalidated.

        Args:
            tc: Temporary code.
            category: Expected code category.
            expected_status: Expected account status.

        Returns:
            The account record.

        Raises:
            gws.ForbiddenError: If no valid account is found.
        """

        try:
            account = self.h.get_account_by_tc(tc, category, expected_status)
        except helper.Error as exc:
            raise gws.ForbiddenError() from exc

        if not account:
            raise gws.ForbiddenError(f'account: {tc=} not found')

        return account

    def mfa_props(self, account: dict, mfa_secret):
        """Create props for the MFA methods available to an account.

        Args:
            account: Account record.
            mfa_secret: MFA secret for the QR codes.

        Returns:
            A list of ``MfaProps``.
        """

        ps = []

        for mo in self.h.mfa_options(account):
            ps.append(MfaProps(
                index=mo.index,
                title=mo.title,
                qrCode=self.h.qr_code_for_mfa(account, mo, mfa_secret)
            ))

        return ps
