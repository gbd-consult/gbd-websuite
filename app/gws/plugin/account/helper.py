from typing import Optional, cast

import gws
import gws.base.action
import gws.base.edit.api as api
import gws.base.edit.helper
import gws.base.feature
import gws.base.layer
import gws.base.legend
import gws.base.model
import gws.base.shape
import gws.base.template
import gws.base.web
import gws.config.util
import gws.gis.crs
import gws.gis.render
import gws.helper.email
import gws.lib.image
import gws.lib.otp
import gws.lib.sa as sa
import gws.lib.jsonx
import gws.lib.mime
import gws.lib.password

from . import core, error

gws.ext.new.helper('account')


class MfaConfig:
    mfaUid: str
    title: str


_DEFAULT_PASSWORD_CREATE_SQL = "crypt( {password}, gen_salt('bf') )"
_DEFAULT_PASSWORD_VERIFY_SQL = "crypt( {password}, {passwordColumn} )"


class MfaOption(gws.Data):
    index: int
    title: str
    adapter: Optional[gws.AuthMultiFactorAdapter]


class Config(gws.Config):
    """Account helper."""

    models: Optional[list[gws.ext.config.model]]
    """Account data models."""
    templates: list[gws.ext.config.template]
    """Templates"""

    loginColumn: str = 'email'

    passwordCreateSql: Optional[str]
    passwordVerifySql: Optional[str]

    tcLifeTime: gws.Duration = 3600

    mfa: Optional[list[MfaConfig]]
    """Multi-factor authentication methods."""

    mfaIssuer: str = ''

    onboardingUrl: str
    """URL for email onboarding."""




class Object(gws.base.edit.helper.Object):
    models: list[gws.Model]
    model: gws.DatabaseModel
    templates: list[gws.Template]

    onboardingUrl: str



    loginColumn: str

    tcLifeTime: int

    mfaOptions: list[MfaOption]
    mfaIssuer: str

    passwordCreateSql: str
    passwordVerifySql: str

    def configure(self):
        self.configure_templates()

        self.configure_models()
        self.model = cast(gws.DatabaseModel, self.models[0])

        self.onboardingUrl = self.cfg('onboardingUrl')

        self.loginColumn = 'email'
        for f in self.model.fields:
            if f.name == core.Columns.username:
                self.loginColumn = f.name

        self.mfaOptions = []
        self.mfaIssuer = self.cfg('mfaIssuer')

        self.passwordCreateSql = self.cfg('passwordCreateSql', default=_DEFAULT_PASSWORD_CREATE_SQL)
        self.passwordVerifySql = self.cfg('passwordVerifySql', default=_DEFAULT_PASSWORD_VERIFY_SQL)

        self.tcLifeTime = self.cfg('tcLifeTime', default=3600)

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self)

    def configure_models(self):
        return gws.config.util.configure_models_for(self)

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
        return [self.model]

    def write_feature(self, req, p):
        f = super().write_feature(req, p)

        if self.onboardingUrl:
            if f and not f.errors and f.isNew:
                account = self.get_account_by_id(f.uid())
                self.send_onboarding_email(account)

        return f

    ##

    def get_account_by_id(self, uid: str) -> Optional[dict]:
        sql = f'''
            SELECT * FROM {self.model.tableName} 
            WHERE {self.model.uidName}=:uid
        '''
        rs = self.model.db.select_text(sql, uid=uid)
        return rs[0] if rs else None

    def get_account_by_credentials(self, credentials: gws.Data) -> Optional[dict]:
        expr = self.passwordVerifySql
        expr = expr.replace('{password}', ':password')
        expr = expr.replace('{passwordColumn}', core.Columns.password)

        username = credentials.get('username')
        password = credentials.get('password')

        sql = f'''
            SELECT 
                {self.model.uidName},
                ( {core.Columns.password} = {expr} ) AS validpassword
            FROM 
                {self.model.tableName}
            WHERE
                {self.loginColumn} = :username
             
        '''
        rs = self.model.db.select_text(sql, username=username, password=password)

        if not rs:
            gws.log.warning(f'get_account_by_credentials: {username=} not found')
            return

        if len(rs) > 1:
            raise error.MultipleEntriesFound(f'get_account_by_credentials: multiple entries for {username=}')

        r = rs[0]
        if not r.get('validpassword'):
            raise error.WrongPassword(f'get_account_by_credentials: {username=} wrong password')

        return self.get_account_by_id(self.get_uid(r))

    def get_account_by_tc(self, tc: str, category: str) -> Optional[dict]:
        sql = f'''
            SELECT 
                {self.model.uidName},
                {core.Columns.tcTime},
                {core.Columns.tcCategory}
            FROM 
                {self.model.tableName} 
            WHERE
                {core.Columns.tc} = :tc
        '''
        rs = self.model.db.select_text(sql, tc=tc)

        if not rs:
            gws.log.warning(f'get_account_by_tc: {tc=} not found')
            return

        self.invalidate_tc(tc)

        if len(rs) > 1:
            raise error.MultipleEntriesFound(f'get_account_by_tc: {tc=} multiple entries')

        r = rs[0]

        if r.get(core.Columns.tcCategory) != category:
            gws.log.warning(f'get_account_by_tc: {category=} {tc=} wrong category')
            return

        if gws.u.stime() - r.get(core.Columns.tcTime, 0) > self.tcLifeTime:
            gws.log.warning(f'get_account_by_tc: {category=} {tc=} expired')
            return

        return self.get_account_by_id(self.get_uid(r))

    ##

    def set_password(self, account: dict, password):
        expr = self.passwordCreateSql
        expr = expr.replace('{password}', ':password')
        expr = expr.replace('{passwordColumn}', core.Columns.password)

        sql = f'''
            UPDATE {self.model.tableName}
            SET
                {core.Columns.password} = {expr}
            WHERE
                {self.model.uidName} = :uid
        '''
        self.model.db.execute_text(sql, password=password, uid=self.get_uid(account))

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
            raise error.InvalidMfaIndex(f'{mfa_option_index=} not found')

        sql = f'''
            UPDATE {self.model.tableName}
            SET
                {core.Columns.mfaUid} = :mfa_uid
            WHERE
                {self.model.uidName} = :uid
        '''
        self.model.db.execute_text(sql, mfa_uid=mfa_uid, uid=self.get_uid(account))

    def mfa_options(self, account: dict) -> list[MfaOption]:
        # @TODO different options per account
        return self.mfaOptions

    def generate_mfa_secret(self, account: dict) -> str:
        secret = gws.lib.otp.random_secret()

        sql = f'''
            UPDATE {self.model.tableName}
            SET
                {core.Columns.mfaSecret} = :secret
            WHERE
                {self.model.uidName} = :uid
        '''
        self.model.db.execute_text(sql, secret=secret, uid=self.get_uid(account))

        return secret

    def qr_code_for_mfa(self, account: dict, mo: MfaOption, secret: str) -> str:
        if not mo.adapter:
            return ''
        url = mo.adapter.key_uri(secret, self.mfaIssuer, account.get(self.loginColumn))
        if not url:
            return ''
        return gws.lib.image.qr_code(url).to_data_url()

        ##

    def set_status(self, account: dict, status: core.Status):
        sql = f'''
            UPDATE {self.model.tableName}
            SET
                {core.Columns.status} = :status
            WHERE
                {self.model.uidName} = :uid
        '''
        self.model.db.execute_text(sql, status=status, uid=self.get_uid(account))

    def reset(self, account: dict):
        sql = f'''
            UPDATE {self.model.tableName}
            SET
                {core.Columns.status} = :status,
                {core.Columns.password} = '',
                {core.Columns.mfaSecret} = ''
            WHERE
                {self.model.uidName} = :uid
        '''
        self.model.db.execute_text(sql, status=core.Status.new, uid=self.get_uid(account))

        if self.onboardingUrl:
            self.send_onboarding_email(account)

    ##

    def send_onboarding_email(self, account: dict):
        tc = self.generate_tc(account, core.Category.onboarding)
        url = self.onboardingUrl + '?onboarding=' + tc
        self.send_mail(account, core.Category.onboarding, {'url': url})

    def generate_tc(self, account: dict, category: str) -> str:
        tc = self.make_tc()

        sql = f'''
            UPDATE {self.model.tableName}
            SET
                {core.Columns.tc} = :tc,
                {core.Columns.tcTime} = :time,
                {core.Columns.tcCategory} = :category
            WHERE
                {self.model.uidName} = :uid
        '''
        self.model.db.execute_text(sql, tc=tc, time=gws.u.stime(), category=category, uid=self.get_uid(account))

        return tc

    def invalidate_tc(self, tc: str):
        sql = f'''
            UPDATE {self.model.tableName}
            SET
                {core.Columns.tc} = '',
                {core.Columns.tcTime} = 0,
                {core.Columns.tcCategory} = ''
            WHERE
                {core.Columns.tc} = :tc
        '''
        self.model.db.execute_text(sql, tc=tc)

    ##

    def get_uid(self, account: dict) -> str:
        return account.get(self.model.uidName)

    def make_tc(self):
        return gws.u.random_string(32)

    def send_mail(self, account: dict, category: str, args: Optional[dict] = None):
        email = account.get('email')
        if not email:
            uid = self.get_uid(account)
            raise error.NoEmail(f'{uid=} has no email')

        args = args or {}
        args['account'] = account

        message = gws.helper.email.Message(
            subject=self.render_template(f'{category}.emailSubject', args),
            mailTo=account.get('email'),
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
