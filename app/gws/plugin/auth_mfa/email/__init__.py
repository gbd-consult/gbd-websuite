"""Multi-factor authenticator that sends TOTPs per email.

The user is required to have an ``email`` attribute.
User secret is dynamically generated each time new TOTP is created.

"""

from typing import Optional, cast

import gws
import gws.base.auth
import gws.plugin.email_helper
import gws.lib.otp

gws.ext.new.authMultiFactorAdapter('email')


class Config(gws.base.auth.mfa.Config):
    templates: Optional[list[gws.ext.config.template]]
    """Email templates."""


class Object(gws.base.auth.mfa.Object):
    templates: list[gws.Template]

    def configure(self):
        self.templates = self.create_children(gws.ext.object.template, self.cfg('templates'))

    def start(self, user):
        if not user.email:
            gws.log.warning(f'email: cannot start, {user.uid=}: no email')
            return
        mfa = super().start(user)
        self.generate_and_send(mfa)
        return mfa

    def verify(self, mfa, payload):
        ok = self.check_totp(mfa, payload['code'])
        return self.verify_attempt(mfa, ok)

    ##

    def generate_and_send(self, mfa: gws.AuthMultiFactorTransaction):
        # NB regenerate secret on each attempt
        mfa.secret = gws.lib.otp.random_secret()

        args = {
            'user': mfa.user,
            'otp': self.generate_totp(mfa),
        }
        message = gws.plugin.email_helper.Message(
            subject=self.render_template('email.subject', args),
            mailTo=mfa.user.email,
            text=self.render_template('email.body', args, mime='text/plain'),
            html=self.render_template('email.body', args, mime='text/html'),
        )

        email_helper = cast(gws.plugin.email_helper.Object, self.root.app.helper('email'))
        email_helper.send_mail(message)

    def render_template(self, subject, args, mime=None):
        tpl = self.root.app.templateMgr.find_template(subject, where=[self], mime=mime)
        if tpl:
            res = tpl.render(gws.TemplateRenderInput(args=args))
            return res.content
        return ''
