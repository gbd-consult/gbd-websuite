"""Multi-factor authentication with a one-time code sent by email.

When a transaction starts, the adapter generates a random secret, creates a
TOTP code from it and sends the code to the ``email`` address of the user. A
restart sends a new code with a new secret. Users without an email address
cannot start a transaction. The client sends the code back as
``{"code": ...}``.

The email subject is rendered from the template ``email.subject``, the body
from the template ``email.body``, once as ``text/plain`` and once as
``text/html``; the HTML part is only added if it is not empty. Templates get
the arguments ``user`` and ``otp``. The mail is sent with the ``email`` helper
(``gws.plugin.email_helper``), which must be configured.

Example::

    auth.mfa+ {
        type "email"
        uid "AUTH_MFA_EMAIL"

        templates+ {
            type "text"
            subject "email.subject"
            text "Your login code"
        }

        templates+ {
            type "text"
            subject "email.body"
            text "Hello {{user.displayName}}, your code is {{otp}}."
        }
    }
"""

from typing import Optional, cast

import gws
import gws.base.auth
import gws.plugin.email_helper
import gws.lib.otp


@gws.ext.config.authMultiFactorAdapter('email')
class Config(gws.base.auth.mfa.Config):
    """Multi-factor authentication with a one-time code sent by email."""

    templates: Optional[list[gws.ext.config.template]]
    """Templates for the email subject and body."""


@gws.ext.object.authMultiFactorAdapter('email')
class Object(gws.base.auth.mfa.Object):
    """Email multi-factor adapter."""

    templates: list[gws.Template]
    """Templates for the email subject and body."""

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
        ok = self.check_totp(mfa, payload.get('code'))
        return self.verify_attempt(mfa, ok)

    ##

    def generate_and_send(self, mfa: gws.AuthMultiFactorTransaction):
        """Generate a new code and send it to the user by email.

        A new random secret is stored in the transaction before the code is
        generated.

        Args:
            mfa: The transaction.

        Raises:
            ``gws.plugin.email_helper.Error``: If the email cannot be sent.
        """
        # NB regenerate secret on each attempt
        mfa.secret = gws.lib.otp.random_secret()

        args = {
            'user': mfa.user,
            'otp': self.generate_totp(mfa),
        }
        message = gws.plugin.email_helper.Message(
            subject=self.render_template('email.subject', args),
            mailTo=mfa.user.email,
            text=self.render_template('email.body', args, mime_type='text/plain'),
            html=self.render_template('email.body', args, mime_type='text/html'),
        )

        email_helper = cast(gws.plugin.email_helper.Object, self.root.app.helper('email'))
        email_helper.send_mail(message)

    def render_template(self, subject, args, mime_type=None):
        """Render a template of the adapter.

        Args:
            subject: Template subject, for example ``email.body``.
            args: Template arguments.
            mime_type: Output mime type of the template to find.

        Returns:
            The rendered content, or an empty string if no template is found.
        """
        tpl = self.root.app.templateMgr.find_template(subject, where=[self], mime_type=mime_type)
        if tpl:
            res = tpl.render(gws.TemplateRenderInput(args=args))
            return res.content
        return ''
