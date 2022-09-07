"""Generate HOTP and TOTP tokens as per rfc4226, rfc6238."""

import base64
import hashlib
import hmac
import random
import time

import gws


class Error(gws.Error):
    pass


def raw_hotp(key_bytes: bytes, counter: int, length: int = 6, digestmod=None) -> str:
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

    digestmod = digestmod or hashlib.sha1
    c = counter.to_bytes(8, byteorder='big')

    hs = hmac.new(key_bytes, c, digestmod).digest()
    offset = hs[-1] & 0xf
    p = hs[offset:offset + 4]
    snum = int.from_bytes(p, byteorder='big', signed=False) & 0x7fffffff
    d = snum % (10 ** length)

    s = str(d)
    while len(s) < length:
        s = '0' + s

    return s


def hotp(base32_secret: str, counter: int, length: int = 6, digestmod=None) -> str:
    try:
        key_bytes = base64.b32decode(base32_secret)
        return raw_hotp(key_bytes, counter, length, digestmod)
    except Exception as exc:
        raise Error() from exc


def totp(base32_secret: str, timestamp: int = None, start: int = 0, step: int = 30, length: int = 6, digestmod=None) -> str:
    if timestamp is None:
        timestamp = int(time.time())
    try:
        counter = (timestamp - start) // step
        key_bytes = base64.b32decode(base32_secret)
        return raw_hotp(key_bytes, counter, length, digestmod)
    except Exception as exc:
        raise Error() from exc


def random_base32_string(length: int = 32) -> str:
    _b32alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'  # from base64.py
    r = random.SystemRandom()
    return ''.join(r.choice(_b32alphabet) for _ in range(length))
