"""Password tools."""

import base64
import hashlib
import hmac
import random


def compare(a: str, b: str) -> bool:
    """Return a == b

        Args:
            a: as String
            b: as String

        Return:
            True if a equals b, False otherwise
    """

    return hmac.compare_digest(a.encode('utf8'), b.encode('utf8'))


def encode(password, algo='sha512'):
    """Encode a password into a hash.

    Args:
          password: String password.
          algo: hashing algorithm. Default is SHA512.

    Return:
        Respective hash value in the format: $algorithm$Salt$hash
    """

    salt = _random_string(8)
    h = _pbkdf2(password, salt, algo)
    return '$'.join(['', algo, salt, base64.urlsafe_b64encode(h).decode('utf8')])


def check(password, encoded):
    """Check if a password matches a hash.

    Args:
         password: String password.
         encoded: Hash of the input password.

    Returns:
        Boolean True if password matches the hash, else False.
    """

    try:
        _, algo, salt, hs = str(encoded).split('$')
        h1 = base64.urlsafe_b64decode(hs)
        h2 = _pbkdf2(password, salt, algo)
    except (TypeError, ValueError):
        return False

    return hmac.compare_digest(h1, h2)


##


def _pbkdf2(password, salt, algo):
    return hashlib.pbkdf2_hmac(algo, password.encode('utf8'), salt.encode('utf8'), 100000)


def _random_string(length):
    a = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    r = random.SystemRandom()
    return ''.join(r.choice(a) for _ in range(length))
