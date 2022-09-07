"""Email sending helper."""

import email.message
import email.policy
import email.utils
import smtplib
import ssl

import gws
import gws.types as t

_MODE_PLAIN = 'plain'
_MODE_SSL = 'ssl'
_MODE_TLS = 'tls'


class SmtpConfig(t.Config):
    mode: str = 'ssl'  #: 'plain', 'ssl' or 'tls'
    host: str  #: hostname
    port: int = 0  #: port
    login: str = ''  #: login
    password: str = ''  #: password
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

        self.mailFrom = self.var('mailFrom')

        self.smtp = self.var('smtp')
        if self.smtp:
            if self.smtp.mode == _MODE_PLAIN:
                self.smtp.port = self.smtp.port or 25
            elif self.smtp.mode == _MODE_SSL:
                self.smtp.port = self.smtp.port or 465
            elif self.smtp.mode == _MODE_TLS:
                self.smtp.port = self.smtp.port or 587
            else:
                raise ValueError(f'invalid smtp mode')

    def send_mail(self, m: Message):
        msg = email.message.EmailMessage(email.policy.EmailPolicy(**_DEFAULT_POLICY))

        msg['Subject'] = m.subject
        msg['To'] = m.mailTo
        msg['From'] = m.mailFrom or self.mailFrom
        msg['Date'] = email.utils.formatdate()
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
        if self.smtp.mode == _MODE_SSL:
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

        if self.smtp.mode == _MODE_TLS:
            srv.starttls(context=ssl.create_default_context())

        if self.smtp.login:
            srv.login(self.smtp.login, self.smtp.password)

        return srv
