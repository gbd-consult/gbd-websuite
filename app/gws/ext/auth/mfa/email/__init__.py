"""Email/OTP two factor auth."""

import gws
import gws.common.auth.mfa
import gws.common.template
import gws.tools.otp
import gws.ext.helper.email
import gws.web.error

import gws.types as t

from gws.common.auth.mfa import Error


class Config(gws.common.auth.mfa.Config):
    pass


class Object(gws.common.auth.mfa.Object):
    def start(self, user):
        if not user.attribute('email'):
            raise Error(f'email required, user={user.uid!r}')

        mf = super().start(user)
        self.generate_and_send(user, mf)

        return mf

    def restart(self, user, mf):
        if not self.is_valid(user, mf):
            raise Error('invalid mfa')

        if mf.restartCount >= self.maxRestarts:
            raise Error('too many restarts')

        mf.restartCount += 1
        self.generate_and_send(user, mf)

    def verify_attempt(self, user, mf, data):
        return self.check_totp(mf, data.token)

    def generate_and_send(self, user, mf):

        # NB regenerate secret on each attempt
        mf.secret = gws.tools.otp.random_base32_string()

        args = {
            'user': user,
            'otp': self.generate_totp(mf),
        }

        message = t.Data(
            mailTo=user.attribute('email'),
            subject=self.render_template('email.subject', args),
            text=self.render_template('email.body', args, mime='text/plain'),
            html=self.render_template('email.body', args, mime='text/html'),
        )

        email_helper = self.root.application.require_helper('email')

        try:
            email_helper.send_mail(message)
        except gws.ext.helper.email.Error as exc:
            gws.log.exception()
            raise Error from exc

    def render_template(self, subject, args, mime=None):
        tpl = gws.common.template.find(self.templates, subject=subject, mime=mime)
        if tpl:
            res = tpl.render(args)
            return res.content
        return ''
