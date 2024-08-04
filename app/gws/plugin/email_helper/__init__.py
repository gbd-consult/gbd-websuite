"""Email sending helper."""

import email.message
import email.policy
import email.utils
import smtplib
import ssl

import gws

gws.ext.new.helper('email')


class SmtpMode(gws.Enum):
    plain = 'plain'
    ssl = 'ssl'
    tls = 'tls'


class SmtpConfig(gws.Config):
    """SMTP server configuration. (added in 8.1)"""

    mode: SmtpMode = 'ssl'
    """Connection mode."""
    host: str
    """SMTP host name"""
    port: int = 0
    """SMTP port."""
    login: str = ''
    """Login"""
    password: str = ''
    """Password."""
    timeout: gws.Duration = 30
    """Connection timeout."""


class Config(gws.Config):
    """Mail helper settings"""

    smtp: SmtpConfig
    """SMTP server configuration."""
    mailFrom: str = ''
    """Default 'From' address."""


class Message(gws.Data):
    """Email message."""

    subject: str
    """Subject."""
    mailTo: str
    """To addresses, comma separated."""
    mailFrom: str
    """From address (default if omitted)."""
    bcc: str
    """Bcc addresses."""
    text: str
    """Plain text content."""
    html: str
    """HTML content."""


class Error(gws.Error):
    pass


##

_DEFAULT_POLICY = {
    'linesep': '\r\n',
    'cte_type': '7bit',
    'utf8': False,
}

_DEFAULT_ENCODING = 'quoted-printable'

_DEFAULT_PORT = {
    SmtpMode.plain: 25,
    SmtpMode.ssl: 465,
    SmtpMode.tls: 587,
}


class _SmtpServer(gws.Data):
    mode: SmtpMode
    host: str
    port: int
    login: str
    password: str
    timeout: int


class Object(gws.Node):
    smtp: _SmtpServer
    mailFrom: str

    def configure(self):
        self.mailFrom = self.cfg('mailFrom')

        p = self.cfg('smtp')

        self.smtp = _SmtpServer(
            mode=p.mode or SmtpMode.ssl,
            host=p.host,
            login=p.login,
            password=p.password,
            timeout=p.timeout,
        )
        self.smtp.port = p.port or _DEFAULT_PORT.get(self.smtp.mode)

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
            # @TODO images
            msg.add_alternative(m.html, subtype='html', cte=_DEFAULT_ENCODING)

        self._send(msg)

    def _send(self, msg):
        if self.smtp:
            try:
                with self._smtp_connection() as conn:
                    conn.send_message(msg)
            except OSError as exc:
                raise Error('SMTP error') from exc

    def _smtp_connection(self):
        if self.smtp.mode == SmtpMode.ssl:
            conn = smtplib.SMTP_SSL(
                host=self.smtp.host,
                port=self.smtp.port,
                timeout=self.smtp.timeout,
                context=ssl.create_default_context(),
            )
        else:
            conn = smtplib.SMTP(
                host=self.smtp.host,
                port=self.smtp.port,
                timeout=self.smtp.timeout,
            )

        # conn.set_debuglevel(2)

        if self.smtp.mode == SmtpMode.tls:
            conn.starttls(context=ssl.create_default_context())

        if self.smtp.login:
            conn.login(self.smtp.login, self.smtp.password)

        return conn
