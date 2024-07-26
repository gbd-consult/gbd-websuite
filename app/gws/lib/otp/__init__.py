"""Generate HOTP and TOTP tokens.

References:
    https://datatracker.ietf.org/doc/html/rfc4226
    https://datatracker.ietf.org/doc/html/rfc6238
"""

from typing import Optional

import base64
import hashlib
import hmac
import random

import gws
import gws.lib.net


class Options(gws.Data):
    start: int
    step: int
    length: int
    tolerance: int
    algo: str


DEFAULTS = Options(
    start=0,
    step=30,
    length=6,
    tolerance=1,
    algo='sha1',
)


def new_hotp(secret: str | bytes, counter: int, options: Optional[Options] = None) -> str:
    """Generate a new HOTP value as per rfc4226 section 5.3."""

    return _raw_otp(_to_bytes(secret), counter, gws.u.merge(DEFAULTS, options))


def new_totp(secret: str | bytes, timestamp: int, options: Optional[Options] = None) -> str:
    """Generate a new TOTP value as per rfc6238 section 4.2."""

    options = gws.u.merge(DEFAULTS, options)
    counter = (timestamp - options.start) // options.step
    return _raw_otp(_to_bytes(secret), counter, options)


def check_totp(input: str, secret: str, timestamp: int, options: Optional[Options] = None) -> bool:
    """Check if the input TOTP is valid.

    Compares the input against several TOTPs within the tolerance window
    ``(timestamp-step*tolerance...timestamp+step*tolerance)``.
    """

    options = gws.u.merge(DEFAULTS, options)

    if len(input) != options.length:
        return False

    for window in range(-options.tolerance, options.tolerance + 1):
        ts = timestamp + options.step * window
        counter = (ts - options.start) // options.step
        totp = _raw_otp(_to_bytes(secret), counter, options)
        gws.log.debug(f'check_totp {timestamp=} {totp=} {input=} {window=}')
        if input == totp:
            return True

    return False


def key_uri(
        method: str,
        secret: str,
        issuer_name: str,
        account_name: str,
        counter: Optional[int] = None,
        options: Optional[Options] = None
) -> str:
    """Create a key uri for auth apps.

    Reference:
        https://github.com/google/google-authenticator/wiki/Key-Uri-Format
    """

    params = {
        'secret': base32_encode(secret),
        'issuer': issuer_name,
    }

    options = gws.u.merge(DEFAULTS, options)

    if options.algo != DEFAULTS.algo:
        params['algorithm'] = options.algo
    if options.length != DEFAULTS.length:
        params['digits'] = options.length
    if options.step != DEFAULTS.step:
        params['period'] = options.step
    if counter is not None:
        params['counter'] = counter

    return 'otpauth://{}/{}:{}?{}'.format(
        method,
        gws.lib.net.quote_param(issuer_name),
        gws.lib.net.quote_param(account_name),
        gws.lib.net.make_qs(params)
    )


def base32_decode(s: str) -> bytes:
    return base64.b32decode(s)


def base32_encode(s: str | bytes) -> str:
    return base64.b32encode(_to_bytes(s)).decode('ascii')


def random_base32_string(length: int = 32) -> str:
    """Generate a random base32 string."""

    if (length & 7) != 0:
        raise ValueError('invalid length')

    r = random.SystemRandom()
    b = r.randbytes((length >> 3) * 5)
    return base32_encode(b)


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
