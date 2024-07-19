"""Generate HOTP and TOTP tokens.

References:
    https://datatracker.ietf.org/doc/html/rfc4226
    https://datatracker.ietf.org/doc/html/rfc6238
"""

import base64
import hashlib
import hmac
import random


def new_hotp(secret: str | bytes, counter: int, length: int = 6, algo: str = 'sha1') -> str:
    """Generate a new HOTP value as per rfc4226 section 5.3."""

    return _raw_otp(_to_bytes(secret), counter, length, algo)


def new_totp(secret: str | bytes, timestamp: int, start: int = 0, step: int = 30, length: int = 6, algo: str = 'sha1') -> str:
    """Generate a new TOTP value as per rfc6238 section 4.2."""

    counter = (timestamp - start) // step
    return _raw_otp(_to_bytes(secret), counter, length, algo)


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

def _raw_otp(key: bytes, counter: int, length: int, algo: str) -> str:
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

    digestmod = getattr(hashlib, algo.lower())
    hs = hmac.new(key, c, digestmod).digest()

    offset = hs[-1] & 0xf
    p = hs[offset:offset + 4]
    snum = int.from_bytes(p, byteorder='big', signed=False) & 0x7fffffff

    d = snum % (10 ** length)

    return f'{d:0{length}d}'


def _to_bytes(s):
    return s.encode('utf8') if isinstance(s, str) else s
