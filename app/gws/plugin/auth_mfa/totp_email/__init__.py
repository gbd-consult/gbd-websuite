"""Multi-factor authenticator that sends TOTPs per email.

The user is required to have an ``email`` attribute.
User secret is dynamically generated each time new TOTP is created.

"""

from typing import Optional, cast

import gws
import gws.base.auth
import gws.helper.email
import gws.lib.otp

gws.ext.new.authMultiFactorAdapter('totp_email')


class Config(gws.base.auth.mfa.Config):
    templates: Optional[list[gws.ext.config.template]]
    """Email templates."""


class Object(gws.base.auth.mfa.Object):
    templates: list[gws.Template]

    def configure(self):
        self.templates = self.create_children(gws.ext.object.template, self.cfg('templates'))

    def start(self, user):
        if not user.email:
            gws.log.warning(f'totp_email: cannot start, {user.uid=}: no email')
            return
        mfa = super().start(user)
        self.generate_and_send(mfa)
        return mfa

    def check_payload(self, mfa, payload):
        return self.check_totp(mfa, payload['code'])

    ##

    def generate_and_send(self, mfa: gws.AuthMultiFactorTransaction):
        # NB regenerate secret on each attempt
        mfa.secret = gws.u.random_string(32)

        args = {
            'user': mfa.user,
            'otp': self.generate_totp(mfa),
        }
        message = gws.helper.email.Message(
            subject=self.render_template('email.subject', args),
            mailTo=mfa.user.email,
            text=self.render_template('email.body', args, mime='text/plain'),
            html=self.render_template('email.body', args, mime='text/html'),
        )

        email_helper = cast(gws.helper.email.Object, self.root.app.helper('email'))
        email_helper.send_mail(message)

    def render_template(self, subject, args, mime=None):
        tpl = self.root.app.templateMgr.find_template(subject, where=[self], mime=mime)
        if tpl:
            res = tpl.render(gws.TemplateRenderInput(args=args))
            return res.content
        return ''
