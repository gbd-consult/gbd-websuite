from typing import Optional, cast

import gws
import gws.base.edit.helper
import gws.config.util
import gws.helper.email
import gws.lib.image
import gws.lib.net
import gws.lib.otp

from . import core

gws.ext.new.helper('account')


class MfaConfig:
    mfaUid: str
    title: str


class Config(gws.Config):
    """Account helper. (added in 8.1)"""

    adminModel: gws.ext.config.model
    """Edit model for account administration."""
    userModel: Optional[gws.ext.config.model]
    """Edit model for end-users accounts."""
    templates: list[gws.ext.config.template]
    """Templates"""

    usernameColumn: str = 'email'
    """Column used as 'login'."""

    passwordCreateSql: Optional[str]
    """SQL expression for computing password hashes."""
    passwordVerifySql: Optional[str]
    """SQL expression for verifying password hashes."""

    tcLifeTime: gws.Duration = 3600
    """Life time for temporary codes."""

    mfa: Optional[list[MfaConfig]]
    """Multi-factor authentication methods the user can choose from."""
    mfaIssuer: str = ''
    """Issuer name for Multi-factor key uris (qr codes)."""

    onboardingUrl: str
    """URL for email onboarding."""
    onboardingCompletionUrl: str = ''
    """URL to redirect after onboarding."""


##


class Error(gws.Error):
    """Account-related error."""
    pass


##


class MfaOption(gws.Data):
    index: int
    title: str
    adapter: Optional[gws.AuthMultiFactorAdapter]


_DEFAULT_PASSWORD_CREATE_SQL = "crypt( {password}, gen_salt('bf') )"
_DEFAULT_PASSWORD_VERIFY_SQL = "crypt( {password}, {passwordColumn} )"


class Object(gws.base.edit.helper.Object):
    adminModel: gws.DatabaseModel
    userModel: gws.DatabaseModel
    templates: list[gws.Template]

    mfaIssuer: str
    mfaOptions: list[MfaOption]
    onboardingUrl: str
    onboardingCompletionUrl: str
    passwordCreateSql: str
    passwordVerifySql: str
    tcLifeTime: int
    usernameColumn: str

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
        f = super().write_feature(req, p)

        if f and not f.errors and f.isNew:
            account = self.get_account_by_id(f.uid())
            self.reset(account)

        return f

    ##

    def get_account_by_id(self, uid: str) -> Optional[dict]:
        sql = f'''
            SELECT * FROM {self.adminModel.tableName}
            WHERE {self.adminModel.uidName}=:uid
        '''
        rs = self.adminModel.db.select_text(sql, uid=uid)
        return rs[0] if rs else None

    def get_account_by_credentials(self, credentials: gws.Data, expected_status: Optional[core.Status] = None) -> Optional[dict]:
        expr = self.passwordVerifySql
        expr = expr.replace('{password}', ':password')
        expr = expr.replace('{passwordColumn}', core.Columns.password)

        username = credentials.get('username')
        password = credentials.get('password')

        sql = f'''
            SELECT
                {self.adminModel.uidName},
                ( {core.Columns.password} = {expr} ) AS validpassword
            FROM
                {self.adminModel.tableName}
            WHERE
                {self.usernameColumn} = :username
        '''
        rs = self.adminModel.db.select_text(sql, username=username, password=password)

        if not rs:
            gws.log.warning(f'get_account_by_credentials: {username=} not found')
            return

        if len(rs) > 1:
            raise Error(f'get_account_by_credentials: multiple entries for {username=}')

        r = rs[0]

        if not r.get('validpassword'):
            raise Error(f'get_account_by_credentials: {username=} wrong password')

        if expected_status:
            status = r.get(core.Columns.status)
            if status != expected_status:
                raise Error(f'get_account_by_credentials: {username=} wrong {status=} {expected_status=}')

        return self.get_account_by_id(self.get_uid(r))

    def get_account_by_tc(self, tc: str, category: str, expected_status: Optional[core.Status] = None) -> Optional[dict]:
        sql = f'''
            SELECT
                {self.adminModel.uidName},
                {core.Columns.tcTime},
                {core.Columns.tcCategory},
                {core.Columns.status}
            FROM
                {self.adminModel.tableName}
            WHERE
                {core.Columns.tc} = :tc
        '''
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

        if expected_status:
            status = r.get(core.Columns.status)
            if status != expected_status:
                gws.log.warning(f'get_account_by_tc: {category=} {tc=} wrong {status=} {expected_status=}')
                return

        return self.get_account_by_id(self.get_uid(r))

    ##

    def set_password(self, account: dict, password):
        expr = self.passwordCreateSql
        expr = expr.replace('{password}', ':password')
        expr = expr.replace('{passwordColumn}', core.Columns.password)

        sql = f'''
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.password} = {expr}
            WHERE
                {self.adminModel.uidName} = :uid
        '''
        self.adminModel.db.execute_text(sql, password=password, uid=self.get_uid(account))

    def validate_password(self, password: str) -> bool:
        if len(password.strip()) == 0:
            return False
        # @TODO password complexity validation
        return True

    ##

    def set_mfa(self, account: dict, mfa_option_index: int):
        mfa_uid = None

        for mo in self.mfa_options(account):
            if mo.index == mfa_option_index:
                mfa_uid = mo.adapter.uid if mo.adapter else ''
                break

        if mfa_uid is None:
            raise Error(f'{mfa_option_index=} not found')

        sql = f'''
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.mfaUid} = :mfa_uid
            WHERE
                {self.adminModel.uidName} = :uid
        '''
        self.adminModel.db.execute_text(sql, mfa_uid=mfa_uid, uid=self.get_uid(account))

    def mfa_options(self, account: dict) -> list[MfaOption]:
        # @TODO different options per account
        return self.mfaOptions

    def generate_mfa_secret(self, account: dict) -> str:
        secret = gws.lib.otp.random_secret()

        sql = f'''
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.mfaSecret} = :secret
            WHERE
                {self.adminModel.uidName} = :uid
        '''
        self.adminModel.db.execute_text(sql, secret=secret, uid=self.get_uid(account))

        return secret

    def qr_code_for_mfa(self, account: dict, mo: MfaOption, secret: str) -> str:
        if not mo.adapter:
            return ''
        url = mo.adapter.key_uri(secret, self.mfaIssuer, account.get(self.usernameColumn))
        if not url:
            return ''
        return gws.lib.image.qr_code(url).to_data_url()

        ##

    def set_status(self, account: dict, status: core.Status):
        sql = f'''
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.status} = :status
            WHERE
                {self.adminModel.uidName} = :uid
        '''
        self.adminModel.db.execute_text(sql, status=status, uid=self.get_uid(account))

    def reset(self, account: dict):
        sql = f'''
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.status} = :status,
                {core.Columns.password} = '',
                {core.Columns.mfaSecret} = ''
            WHERE
                {self.adminModel.uidName} = :uid
        '''
        self.adminModel.db.execute_text(sql, status=core.Status.new, uid=self.get_uid(account))

        if self.onboardingUrl:
            self.send_onboarding_email(account)

    ##

    def send_onboarding_email(self, account: dict):
        tc = self.generate_tc(account, core.Category.onboarding)
        url = gws.lib.net.add_params(self.onboardingUrl, onboarding=tc)
        self.send_mail(account, core.Category.onboarding, {'url': url})

    def generate_tc(self, account: dict, category: str) -> str:
        tc = self.make_tc()

        sql = f'''
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.tc} = :tc,
                {core.Columns.tcTime} = :time,
                {core.Columns.tcCategory} = :category
            WHERE
                {self.adminModel.uidName} = :uid
        '''
        self.adminModel.db.execute_text(sql, tc=tc, time=gws.u.stime(), category=category, uid=self.get_uid(account))

        return tc

    def clear_tc(self, account: dict):
        sql = f'''
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.tc} = '',
                {core.Columns.tcTime} = 0,
                {core.Columns.tcCategory} = ''
            WHERE
                {self.adminModel.uidName} = :uid
        '''
        self.adminModel.db.execute_text(sql, uid=self.get_uid(account))

    def invalidate_tc(self, tc: str):
        sql = f'''
            UPDATE {self.adminModel.tableName}
            SET
                {core.Columns.tc} = '',
                {core.Columns.tcTime} = 0,
                {core.Columns.tcCategory} = ''
            WHERE
                {core.Columns.tc} = :tc
        '''
        self.adminModel.db.execute_text(sql, tc=tc)

    ##

    def get_uid(self, account: dict) -> str:
        return account.get(self.adminModel.uidName)

    def make_tc(self):
        return gws.u.random_string(32)

    def send_mail(self, account: dict, category: str, args: Optional[dict] = None):
        email = account.get(core.Columns.email)
        if not email:
            raise Error(f'account {self.get_uid(account)}: no email')

        args = args or {}
        args['account'] = account

        message = gws.helper.email.Message(
            subject=self.render_template(f'{category}.emailSubject', args),
            mailTo=email,
            text=self.render_template(f'{category}.emailBody', args, mime='text/plain'),
            html=self.render_template(f'{category}.emailBody', args, mime='text/html'),
        )

        email_helper = cast(gws.helper.email.Object, self.root.app.helper('email'))
        email_helper.send_mail(message)

    def render_template(self, subject, args, mime=None):
        tpl = self.root.app.templateMgr.find_template(subject, where=[self], mime=mime)
        if tpl:
            res = tpl.render(gws.TemplateRenderInput(args=args))
            return res.content
        return ''
