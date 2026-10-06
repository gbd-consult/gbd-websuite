"""Account helper."""

from typing import Optional, cast

import gws
import gws.base.edit.helper
import gws.config.util
import gws.plugin.email_helper
import gws.lib.image
import gws.lib.net
import gws.lib.otp

from . import core


class MfaConfig(gws.Config):
    """Multi-factor authentication method offered to users."""

    mfaUid: str
    """UID of the multi-factor authentication adapter."""
    title: str
    """Title of the method shown to users."""


@gws.ext.config.helper('account')
class Config(gws.Config):
    """Accounts table, onboarding and password settings, shared by account components."""

    adminModel: gws.ext.config.model
    """Model of the accounts table for account administrators."""
    userModel: Optional[gws.ext.config.model]
    """Model for account data that users can edit themselves."""
    templates: list[gws.ext.config.template]
    """Templates for account emails."""

    usernameColumn: str = 'email'
    """Column of the accounts table used as the login name."""

    passwordCreateSql: Optional[str]
    """SQL expression for computing password hashes."""
    passwordVerifySql: Optional[str]
    """SQL expression for checking a password against the stored hash."""

    tcLifeTime: gws.Duration = '3600'
    """Validity period of temporary codes."""

    mfa: Optional[list[MfaConfig]]
    """Multi-factor authentication methods the user can choose from."""
    mfaIssuer: str = ''
    """Issuer name for multi-factor key URIs (QR codes)."""

    onboardingUrl: str
    """Onboarding page URL sent to new users."""
    onboardingCompletionUrl: str = ''
    """URL to redirect to after onboarding."""


##


class Error(gws.Error):
    """Account-related error."""

    pass


##


class MfaOption(gws.Data):
    """Configured MFA method."""

    index: int
    """Position in the ``mfa`` configuration list, starting with 1."""
    title: str
    """Title shown to users."""
    adapter: Optional[gws.AuthMultiFactorAdapter]
    """MFA adapter, ``None`` for the "no MFA" option."""


_DEFAULT_PASSWORD_CREATE_SQL = "crypt( {password}, gen_salt('bf') )"
_DEFAULT_PASSWORD_VERIFY_SQL = 'crypt( {password}, {passwordColumn} )'


@gws.ext.object.helper('account')
class Object(gws.base.edit.helper.Object):
    """Account helper.

    Holds the account configuration and implements account queries and updates in the table of ``adminModel``,
    temporary codes, passwords, MFA and emails. As an edit helper, it serves the edit API
    of the ``accountadmin`` action for ``adminModel``.
    """

    adminModel: gws.DatabaseModel
    """Model of the accounts table for administrators."""
    userModel: gws.DatabaseModel
    """Model for account data that users can edit themselves, not created yet."""
    templates: list[gws.Template]
    """Email templates."""

    mfaIssuer: str
    """Issuer name for MFA key URIs."""
    mfaOptions: list[MfaOption]
    """Configured MFA methods."""
    onboardingUrl: str
    """Onboarding page URL sent to new users."""
    onboardingCompletionUrl: str
    """URL to go to after the onboarding, defaults to ``onboardingUrl``."""
    passwordCreateSql: str
    """SQL expression that computes a password hash, with the placeholders ``{password}`` and ``{passwordColumn}``."""
    passwordVerifySql: str
    """SQL expression that computes the hash to compare with the stored one, with the same placeholders."""
    tcLifeTime: int
    """Validity period of temporary codes in seconds."""
    usernameColumn: str
    """Column with the login name."""

    def configure(self):
        self.configure_templates()

        self.adminModel = cast(gws.DatabaseModel, self.create_child(gws.ext.object.model, self.cfg('adminModel')))

        self.mfaIssuer = self.cfg('mfaIssuer')
        self.mfaOptions = []

        self.onboardingUrl = self.cfg('onboardingUrl')
        self.onboardingCompletionUrl = self.cfg('onboardingCompletionUrl') or self.onboardingUrl

        self.passwordCreateSql = self.cfg('passwordCreateSql', default=_DEFAULT_PASSWORD_CREATE_SQL)
        self.passwordVerifySql = self.cfg('passwordVerifySql', default=_DEFAULT_PASSWORD_VERIFY_SQL)

        self.tcLifeTime = self.cfg('tcLifeTime', default=3600)

        self.usernameColumn = self.cfg('usernameColumn', default=core.Columns.email)

    def configure_templates(self):
        """Configure the email templates.

        Returns:
            ``True`` if templates were configured.
        """


        return gws.config.util.configure_templates_for(self)

    def post_configure(self):
        for n, c in enumerate(self.cfg('mfa', default=[]), 1):
            opt = MfaOption(index=n, title=c.title)
            if c.mfaUid:
                opt.adapter = self.root.get(c.mfaUid)
                if not opt.adapter:
                    raise gws.ConfigurationError(f'MFA Adapter not found {c.mfaUid=}')
            self.mfaOptions.append(opt)

    ##

    def get_models(self, req, p):
        return [self.adminModel]

    def write_feature(self, req, p):
        is_new = p.feature.isNew
        f = super().write_feature(req, p)

        if f and not f.errors and is_new:
            account = self.get_account_by_id(f.uid())
            self.reset(account)

        return f

    ##

    def get_account_by_id(self, uid: str) -> Optional[dict]:
        """Find an account by its primary key.

        Args:
            uid: Primary key value.

        Returns:
            The account record, or ``None`` if not found.
        """


        sql = f"""
            SELECT * FROM {self.adminModel.tableName}
            WHERE {self.adminModel.uidName}=:uid
        """
        rs = self.adminModel.db.select_text(sql, uid=uid)
        return rs[0] if rs else None

    def get_account_by_credentials(self, credentials: gws.Data, expected_status: Optional[core.Status] = None) -> Optional[dict]:
        """Find an account by login name and password.

        Args:
            credentials: Credentials with ``username`` and ``password``.
            expected_status: If given, the account must have this status.

        Returns:
            The account record, or ``None`` if the login name is not found.

        Raises:
            Error: If the login name is not unique, the password is wrong, or the status is not the expected one.
        """


        expr = self.passwordVerifySql
        expr = expr.replace('{password}', ':password')
        expr = expr.replace('{passwordColumn}', core.Columns.password)

        username = credentials.get('username')
        password = credentials.get('password')

        sql = f"""
            SELECT
                {self.adminModel.uidName},
                ( {core.Columns.password} = {expr} ) AS validpassword,
                {core.Columns.status}
            FROM
                {self.adminModel.tableName}
            WHERE
                {self.usernameColumn} = :username
        """
        rs = self.adminModel.db.select_text(sql, username=username, password=password)

        if not rs:
            gws.log.warning(f'get_account_by_credentials: {username=} not found')
            return

        if len(rs) > 1:
            raise Error(f'get_account_by_credentials: multiple entries for {username=}')

        r = rs[0]

        if not r.get('validpassword'):
            raise Error(f'get_account_by_credentials: {username=} wrong password')

        if expected_status is not None:
            status = r.get(core.Columns.status)
            if status != expected_status:
                raise Error(f'get_account_by_credentials: {username=} wrong {status=} {expected_status=}')

        return self.get_account_by_id(self.get_uid(r))

    def get_account_by_tc(self, tc: str, category: str, expected_status: Optional[core.Status] = None) -> Optional[dict]:
        """Find an account by a temporary code.

        If the code is found, it is invalidated, so that it can be used only once.

        Args:
            tc: Temporary code.
            category: Expected category of the code.
            expected_status: If given, the account must have this status.

        Returns:
            The account record, or ``None`` if the code is not found, has a different category, is expired,
            or the account has a different status.

        Raises:
            Error: If the code is not unique.
        """


        sql = f"""
            SELECT
                {self.adminModel.uidName},
                {core.Columns.tcTime},
                {core.Columns.tcCategory},
                {core.Columns.status}
            FROM
                {self.adminModel.tableName}
            WHERE
                {core.Columns.tc} = :tc
        """
        rs = self.adminModel.db.select_text(sql, tc=tc)

        if not rs:
            gws.log.warning(f'get_account_by_tc: {tc=} not found')
            return

        self.invalidate_tc(tc)

        if len(rs) > 1:
            raise Error(f'get_account_by_tc: {tc=} multiple entries')

        r = rs[0]

        if r.get(core.Columns.tcCategory) != category:
            gws.log.warning(f'get_account_by_tc: {category=} {tc=} wrong category')
            return

        if gws.u.stime() - r.get(core.Columns.tcTime, 0) > self.tcLifeTime:
            gws.log.warning(f'get_account_by_tc: {category=} {tc=} expired')
            return

        if expected_status is not None:
            status = r.get(core.Columns.status)
            if status != expected_status:
                gws.log.warning(f'get_account_by_tc: {category=} {tc=} wrong {status=} {expected_status=}')
                return

        return self.get_account_by_id(self.get_uid(r))

    ##

    def set_password(self, account: dict, password):
        """Store the hash of a new password.

        Args:
            account: Account record.
            password: New password in plain text.
        """


        expr = self.passwordCreateSql
        expr = expr.replace('{password}', ':password')
        expr = expr.replace('{passwordColumn}', core.Columns.password)

        sql = f"""
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.password} = {expr}
            WHERE
                {self.adminModel.uidName} = :uid
        """
        self.adminModel.db.execute_text(sql, password=password, uid=self.get_uid(account))

    def validate_password(self, password: str) -> bool:
        """Check if a password is acceptable.

        Currently only empty passwords are rejected.

        Args:
            password: Password in plain text.

        Returns:
            ``True`` if the password is acceptable.
        """


        if len(password.strip()) == 0:
            return False
        # @TODO password complexity validation
        return True

    ##

    def set_mfa(self, account: dict, mfa_option_index: int):
        """Store the selected MFA method of an account.

        Args:
            account: Account record.
            mfa_option_index: Index of the MFA option.

        Raises:
            Error: If there is no option with this index.
        """


        mfa_uid = None

        for mo in self.mfa_options(account):
            if mo.index == mfa_option_index:
                mfa_uid = mo.adapter.uid if mo.adapter else ''
                break

        if mfa_uid is None:
            raise Error(f'{mfa_option_index=} not found')

        sql = f"""
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.mfaUid} = :mfa_uid
            WHERE
                {self.adminModel.uidName} = :uid
        """
        self.adminModel.db.execute_text(sql, mfa_uid=mfa_uid, uid=self.get_uid(account))

    def mfa_options(self, account: dict) -> list[MfaOption]:
        """Get the MFA methods available to an account.

        Currently all configured methods are available to all accounts.

        Args:
            account: Account record.

        Returns:
            A list of MFA options.
        """


        # @TODO different options per account
        return self.mfaOptions

    def generate_mfa_secret(self, account: dict) -> str:
        """Generate and store a new MFA secret.

        Args:
            account: Account record.

        Returns:
            The secret.
        """


        secret = gws.lib.otp.random_secret()

        sql = f"""
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.mfaSecret} = :secret
            WHERE
                {self.adminModel.uidName} = :uid
        """
        self.adminModel.db.execute_text(sql, secret=secret, uid=self.get_uid(account))

        return secret

    def qr_code_for_mfa(self, account: dict, mo: MfaOption, secret: str) -> str:
        """Create a QR code of the key URI for an MFA method.

        Args:
            account: Account record.
            mo: MFA option.
            secret: MFA secret.

        Returns:
            The QR code image as a data URL, or an empty string if the method has no adapter or no key URI.
        """


        if not mo.adapter:
            return ''
        url = mo.adapter.key_uri(secret, self.mfaIssuer, account.get(self.usernameColumn))
        if not url:
            return ''
        return gws.lib.image.qr_code(url).to_data_url()

        ##

    def set_status(self, account: dict, status: core.Status):
        """Set the status of an account.

        Args:
            account: Account record.
            status: New status.
        """


        sql = f"""
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.status} = :status
            WHERE
                {self.adminModel.uidName} = :uid
        """
        self.adminModel.db.execute_text(sql, status=status, uid=self.get_uid(account))

    def reset(self, account: dict):
        """Reset an account.

        Clears the password and the MFA secret and sets the status to ``new``.
        If ``onboardingUrl`` is configured, the onboarding email is sent.

        Args:
            account: Account record.

        Raises:
            Error: If the onboarding email is due and the account has no email address.
        """


        sql = f"""
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.status} = :status,
                {core.Columns.password} = '',
                {core.Columns.mfaSecret} = ''
            WHERE
                {self.adminModel.uidName} = :uid
        """
        self.adminModel.db.execute_text(sql, status=core.Status.new, uid=self.get_uid(account))

        if self.onboardingUrl:
            self.send_onboarding_email(account)

    ##

    def send_onboarding_email(self, account: dict):
        """Generate an onboarding code and send the onboarding email with the link.

        Args:
            account: Account record.

        Raises:
            Error: If the account has no email address.
        """


        tc = self.generate_tc(account, core.Category.onboarding)
        url = gws.lib.net.add_params(self.onboardingUrl, onboarding=tc)
        self.send_mail(account, core.Category.onboarding, {'url': url})

    def generate_tc(self, account: dict, category: str) -> str:
        """Generate and store a new temporary code.

        Args:
            account: Account record.
            category: Code category.

        Returns:
            The code.
        """


        tc = self.make_tc()

        sql = f"""
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.tc} = :tc,
                {core.Columns.tcTime} = :time,
                {core.Columns.tcCategory} = :category
            WHERE
                {self.adminModel.uidName} = :uid
        """
        self.adminModel.db.execute_text(sql, tc=tc, time=gws.u.stime(), category=category, uid=self.get_uid(account))

        return tc

    def clear_tc(self, account: dict):
        """Remove the temporary code of an account.

        Args:
            account: Account record.
        """


        sql = f"""
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.tc} = '',
                {core.Columns.tcTime} = 0,
                {core.Columns.tcCategory} = ''
            WHERE
                {self.adminModel.uidName} = :uid
        """
        self.adminModel.db.execute_text(sql, uid=self.get_uid(account))

    def invalidate_tc(self, tc: str):
        """Remove a temporary code from all accounts that have it.

        Args:
            tc: Temporary code.
        """


        sql = f"""
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.tc} = '',
                {core.Columns.tcTime} = 0,
                {core.Columns.tcCategory} = ''
            WHERE
                {core.Columns.tc} = :tc
        """
        self.adminModel.db.execute_text(sql, tc=tc)

    ##

    def get_uid(self, account: dict) -> str:
        """Get the primary key of an account.

        Args:
            account: Account record.

        Returns:
            The primary key value.
        """


        return account.get(self.adminModel.uidName)

    def make_tc(self):
        """Create a random temporary code.

        Returns:
            A random string of 32 characters.
        """


        return gws.u.random_string(32)

    def send_mail(self, account: dict, category: str, args: Optional[dict] = None):
        """Send an email to an account.

        The subject and the body are rendered from the templates ``<category>.emailSubject`` and
        ``<category>.emailBody`` (plain text and HTML), and sent with the ``email`` helper.

        Args:
            account: Account record.
            category: Email category, see ``core.Category``.
            args: Template arguments, the account record is added as ``account``.

        Raises:
            Error: If the account has no email address.
        """


        email = account.get(core.Columns.email)
        if not email:
            raise Error(f'account {self.get_uid(account)}: no email')

        args = args or {}
        args['account'] = account

        message = gws.plugin.email_helper.Message(
            subject=self.render_template(f'{category}.emailSubject', args),
            mailTo=email,
            text=self.render_template(f'{category}.emailBody', args, mime_type='text/plain'),
            html=self.render_template(f'{category}.emailBody', args, mime_type='text/html'),
        )

        email_helper = cast(gws.plugin.email_helper.Object, self.root.app.helper('email'))
        email_helper.send_mail(message)

    def render_template(self, subject, args, mime_type=None):
        """Render a template of this helper.

        Args:
            subject: Template subject.
            args: Template arguments.
            mime_type: Template mime type.

        Returns:
            The rendered content, or an empty string if there is no such template.
        """


        tpl = self.root.app.templateMgr.find_template(subject, where=[self], mime_type=mime_type)
        if tpl:
            res = tpl.render(gws.TemplateRenderInput(args=args))
            return res.content
        return ''
