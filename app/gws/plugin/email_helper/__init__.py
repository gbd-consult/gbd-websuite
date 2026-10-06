"""Email helper.

The ``email`` helper sends email messages over SMTP. A ``Message`` has a
subject, recipients, an optional sender and Bcc addresses, a plain text part
and an optional HTML part. Both parts are encoded as quoted-printable. If the
message has no sender, the configured ``mailFrom`` is used.

The SMTP connection mode is ``ssl`` (SMTP over SSL, default port 465),
``tls`` (STARTTLS, default port 587) or ``plain`` (no encryption, default
port 25). If ``login`` is set, the helper logs in before sending.

The helper is used, for example, by the email multi-factor adapter
(``gws.plugin.auth_mfa.email``) and the account plugin. It must be configured
with an SMTP server.

Example::

    helpers+ {
        type "email"
        mailFrom "gws@example.com"
        smtp {
            host "smtp.example.com"
            mode "tls"
            login "gws"
            password "secret"
        }
    }

Usage in Python::

    helper = cast(gws.plugin.email_helper.Object, root.app.helper('email'))
    helper.send_mail(gws.plugin.email_helper.Message(
        subject='Hello',
        mailTo='user_1@example.com',
        text='Hello, world.',
    ))
"""

import email.message
import email.policy
import email.utils
import smtplib
import ssl

import gws


class SmtpMode(gws.Enum):
    """SMTP connection modes."""
    plain = 'plain'
    """Plain connection without encryption."""
    ssl = 'ssl'
    """SSL connection."""
    tls = 'tls'
    """TLS connection (STARTTLS)."""


class SmtpConfig(gws.Config):
    """SMTP server configuration."""

    mode: SmtpMode = SmtpMode.ssl
    """Connection encryption mode."""
    host: str
    """SMTP host name."""
    port: int = 0
    """SMTP port."""
    login: str = ''
    """Login name for the SMTP server."""
    password: str = ''
    """Password for the SMTP server."""
    timeout: gws.Duration = '30'
    """Connection timeout."""


@gws.ext.config.helper('email')
class Config(gws.Config):
    """Sending of emails via SMTP."""

    smtp: SmtpConfig
    """SMTP server configuration."""
    mailFrom: str = ''
    """Sender address for messages that do not set one."""


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
    """Raised when an email cannot be sent."""

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
    """SMTP server settings."""

    mode: SmtpMode
    """Connection mode."""
    host: str
    """Host name."""
    port: int
    """Port."""
    login: str
    """Login name, empty for no login."""
    password: str
    """Password."""
    timeout: int
    """Connection timeout in seconds."""


@gws.ext.object.helper('email')
class Object(gws.Node):
    """Email helper."""

    smtp: _SmtpServer
    """SMTP server settings."""
    mailFrom: str
    """Default sender address."""

    def configure(self):
        self.mailFrom = self.cfg('mailFrom')

        p = self.cfg('smtp')
        if not p:
            raise gws.ConfigurationError('email helper: smtp is not configured')

        self.smtp = _SmtpServer(
            mode=p.mode or SmtpMode.ssl,
            host=p.host,
            login=p.login,
            password=p.password,
            timeout=p.timeout,
        )
        self.smtp.port = p.port or _DEFAULT_PORT.get(self.smtp.mode)

    def send_mail(self, m: Message):
        """Send an email message.

        Args:
            m: The message.

        Raises:
            ``Error``: If the connection to the SMTP server or the sending fails.
        """
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
        """Send a message with the SMTP server."""
        try:
            with self._smtp_connection() as conn:
                conn.send_message(msg)
        except OSError as exc:
            raise Error('SMTP error') from exc

    def _smtp_connection(self):
        """Open an SMTP connection and log in, if configured."""
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
