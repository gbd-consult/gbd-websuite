"""Email sending helper."""

import smtplib
import ssl
import email.message
import email.policy

import gws
import gws.types as t


# @TODO starttls

class SmtpConfig(t.Config):
    host: str  #: hostname
    port: int  #: port
    login: str = ''  #: login
    password: str = ''  #: password
    secure: bool = True  #: use SSL
    timeout: t.Duration = 30  #: connection timeout


class Config(t.WithType):
    """Mail server settings"""

    smtp: t.Optional[SmtpConfig]  #: SMTP server configuration
    mailFrom: str = ''  #: default From address


class Message(t.Data):
    subject: str
    mailTo: str
    mailFrom: str
    bcc: str
    text: str
    html: str


class Error(gws.Error):
    pass


_DEFAULT_POLICY = {
    'linesep': '\r\n',
    'cte_type': '7bit',
    'utf8': False,
}

_DEFAULT_ENCODING = 'quoted-printable'


class Object(gws.Object):
    def configure(self):
        super().configure()

        self.smtp = self.var('smtp')
        self.mailFrom = self.var('mailFrom')

    def send_mail(self, m: Message):
        msg = email.message.EmailMessage(email.policy.EmailPolicy(**_DEFAULT_POLICY))

        msg['Subject'] = m.subject
        msg['To'] = m.mailTo
        msg['From'] = m.mailFrom or self.mailFrom

        if m.bcc:
            msg['Bcc'] = m.bcc

        msg.set_content(m.text, cte=_DEFAULT_ENCODING)
        if m.html:
            msg.add_alternative(m.html, subtype='html', cte=_DEFAULT_ENCODING)

        self._send(msg)

    def _send(self, msg):
        if self.smtp:
            try:
                with self._smtp_connection() as srv:
                    srv.send_message(msg)
            except OSError as exc:
                raise Error() from exc

    def _smtp_connection(self):
        if self.smtp.secure:
            srv = smtplib.SMTP_SSL(
                host=self.smtp.host,
                port=self.smtp.port,
                timeout=self.smtp.timeout,
                context=ssl.create_default_context(),
            )
        else:
            srv = smtplib.SMTP(
                host=self.smtp.host,
                port=self.smtp.port,
                timeout=self.smtp.timeout,
            )

        if self.root.application.developer_option('smtp.debug'):
            srv.set_debuglevel(2)

        if self.smtp.login:
            srv.login(self.smtp.login, self.smtp.password)

        return srv
