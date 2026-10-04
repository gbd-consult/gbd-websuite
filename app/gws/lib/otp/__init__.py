"""Generate and check HOTP and TOTP one-time passwords.

This package implements HMAC-based (HOTP, RFC 4226) and time-based (TOTP, RFC 6238)
one-time passwords, as used by multi-factor authentication. It also creates ``otpauth://``
key URIs for authenticator apps and random secrets.

All functions accept an optional ``Options`` object. Options that are not set are taken
from ``DEFAULTS`` (30 second step, 6 digits, SHA-1, tolerance of one step).

Example::

    import time
    import gws.lib.otp

    secret = gws.lib.otp.random_secret()
    uri = gws.lib.otp.totp_key_uri(secret, 'GWS', 'user@example.com')
    ok = gws.lib.otp.check_totp(user_input, secret, int(time.time()))

References:
    https://datatracker.ietf.org/doc/html/rfc4226
    https://datatracker.ietf.org/doc/html/rfc6238
    https://github.com/google/google-authenticator/wiki/Key-Uri-Format
"""

from typing import Optional, cast

import base64
import hashlib
import hmac
import random

import gws
import gws.lib.net


class Options(gws.Data):
    """OTP generation options."""

    start: int
    """Start time (Unix timestamp) for TOTP counting."""
    step: int
    """TOTP time step in seconds."""
    length: int
    """Number of digits in a token."""
    tolerance: int
    """Number of time steps before and after the current one that are also accepted."""
    algo: str
    """Hash algorithm name, as in ``hashlib``, e.g. ``sha1``."""


DEFAULTS = Options(
    start=0,
    step=30,
    length=6,
    tolerance=1,
    algo='sha1',
)


def new_hotp(secret: str | bytes, counter: int, options: Optional[Options] = None) -> str:
    """Generate an HOTP token as per RFC 4226 section 5.3.

    Args:
        secret: Shared secret.
        counter: Counter value.
        options: Generation options.

    Returns:
        The token as a string of digits.
    """

    options = cast(Options, gws.u.merge(DEFAULTS, options))
    return _raw_otp(_to_bytes(secret), counter, options)


def new_totp(secret: str | bytes, timestamp: int, options: Optional[Options] = None) -> str:
    """Generate a TOTP token as per RFC 6238 section 4.2.

    Args:
        secret: Shared secret.
        timestamp: Unix timestamp.
        options: Generation options.

    Returns:
        The token as a string of digits.
    """

    options = cast(Options, gws.u.merge(DEFAULTS, options))
    counter = (timestamp - options.start) // options.step
    return _raw_otp(_to_bytes(secret), counter, options)


def check_totp(input: str, secret: str, timestamp: int, options: Optional[Options] = None) -> bool:
    """Check if a TOTP token is valid.

    Compares the input against the TOTP tokens within the tolerance window
    ``(timestamp-step*tolerance...timestamp+step*tolerance)``.

    Args:
        input: Token entered by the user.
        secret: Shared secret.
        timestamp: Unix timestamp.
        options: Generation options.

    Returns:
        ``True`` if the input matches one of the tokens in the window.
    """

    options = cast(Options, gws.u.merge(DEFAULTS, options))

    if len(input) != options.length:
        return False

    ok = False

    for window in range(-options.tolerance, options.tolerance + 1):
        ts = timestamp + options.step * window
        counter = (ts - options.start) // options.step
        totp = _raw_otp(_to_bytes(secret), counter, options)
        if hmac.compare_digest(_to_bytes(input), _to_bytes(totp)):
            ok = True

    return ok


def totp_key_uri(
        secret: str | bytes,
        issuer_name: str,
        account_name: str,
        options: Optional[Options] = None
) -> str:
    """Create a TOTP key URI for authenticator apps.

    Args:
        secret: Shared secret, encoded as base32 in the URI.
        issuer_name: Issuer name, e.g. the application name.
        account_name: Account name, e.g. the user login.
        options: Generation options. Only non-default values are included in the URI.

    Returns:
        An ``otpauth://totp/...`` URI.
    """
    return _key_uri('totp', secret, issuer_name, account_name, None, options)


def hotp_key_uri(
        secret: str | bytes,
        issuer_name: str,
        account_name: str,
        counter: int,
        options: Optional[Options] = None
) -> str:
    """Create an HOTP key URI for authenticator apps.

    Args:
        secret: Shared secret, encoded as base32 in the URI.
        issuer_name: Issuer name, e.g. the application name.
        account_name: Account name, e.g. the user login.
        counter: Initial counter value.
        options: Generation options. Only non-default values are included in the URI.

    Returns:
        An ``otpauth://hotp/...`` URI.
    """
    return _key_uri('hotp', secret, issuer_name, account_name, counter, options)


def _key_uri(
        method: str,
        secret: str | bytes,
        issuer_name: str,
        account_name: str,
        counter: Optional[int] = None,
        options: Optional[Options] = None
) -> str:
    """Create a key URI for authenticator apps (Google Authenticator Key Uri Format)."""

    params: dict = {
        'secret': base32_encode(secret),
        'issuer': issuer_name,
    }

    options = cast(Options, gws.u.merge(DEFAULTS, options))

    if options.algo != DEFAULTS.algo:
        params['algorithm'] = options.algo
    if options.length != DEFAULTS.length:
        params['digits'] = options.length

    if method == 'hotp':
        params['counter'] = counter
    elif options.step != DEFAULTS.step:
        params['period'] = options.step

    return 'otpauth://{}/{}:{}?{}'.format(
        method,
        gws.lib.net.quote_param(issuer_name),
        gws.lib.net.quote_param(account_name),
        gws.lib.net.make_qs(params)
    )


def base32_decode(s: str) -> bytes:
    """Decode a base32 string.

    Args:
        s: Base32 string.

    Returns:
        Decoded bytes.
    """
    return base64.b32decode(s)


def base32_encode(s: str | bytes) -> str:
    """Encode a string or bytes as base32.

    Args:
        s: Value to encode. Strings are encoded as UTF-8 first.

    Returns:
        Base32 string.
    """
    return base64.b32encode(_to_bytes(s)).decode('ascii')


def random_secret(base32_length: int = 32) -> str:
    """Generate a random secret of printable ASCII characters.

    The secret length is chosen so that its base32 encoding is exactly ``base32_length`` characters long.

    Args:
        base32_length: Length of the base32-encoded secret, must be a multiple of 8.

    Returns:
        The secret.

    Raises:
        ``ValueError``: If ``base32_length`` is not a multiple of 8.
    """

    if (base32_length & 7) != 0:
        raise ValueError('invalid length')

    size = (base32_length >> 3) * 5
    r = random.SystemRandom()
    return ''.join(chr(r.randint(0x21, 0x7f)) for _ in range(size))


##

def _raw_otp(key: bytes, counter: int, options: Options) -> str:
    # https://www.rfc-editor.org/rfc/rfc4226#section-5.3
    #
    # Step 1: Generate an HMAC-SHA-1 value
    # Let HS = HMAC-SHA-1(K,C)  // HS is a 20-byte string
    #
    # Step 2: Generate a 4-byte string (Dynamic Truncation)
    # Let Sbits = DT(HS)   //  DT, defined below, returns a 31-bit string
    #
    #   Let OffsetBits be the low-order 4 bits of String[19]
    #   Offset = StToNum(OffsetBits) // 0 <= OffSet <= 15
    #   Let P = String[OffSet]...String[OffSet+3]
    #   Return the Last 31 bits of P
    #
    # Let Snum  = StToNum(Sbits)   // Convert S to a number in 0...2^{31}-1
    #
    # Step 3: Compute an HOTP value
    # Return D = Snum mod 10^Digit //  D is a number in the range 0...10^{Digit}-1

    c = counter.to_bytes(8, byteorder='big')

    digestmod = getattr(hashlib, options.algo.lower())
    hs = hmac.new(key, c, digestmod).digest()

    offset = hs[-1] & 0xf
    p = hs[offset:offset + 4]
    snum = int.from_bytes(p, byteorder='big', signed=False) & 0x7fffffff

    d = snum % (10 ** options.length)

    return f'{d:0{options.length}d}'


def _to_bytes(s):
    return s.encode('utf8') if isinstance(s, str) else s


def _option(options, key, default):
    if not options:
        return default
    return getattr(options, key, default)
