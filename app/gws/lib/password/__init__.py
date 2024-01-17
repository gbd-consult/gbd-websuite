"""Password tools."""

import base64
import hashlib
import hmac
import random


def compare(a: str, b: str) -> bool:
    """Compares two Strings in a safe way to prevent timing attacks.

        Args:
            a: 1st String.
            b: 2nd String.

        Returns:
            ``True`` if a equals b, ``False`` otherwise.
    """

    return hmac.compare_digest(a.encode('utf8'), b.encode('utf8'))


def encode(password: str, algo: str = 'sha512') -> str:
    """Encode a password into a hash.

    Args:
          password: String password.
          algo: Hashing algorithm. Default is SHA512.

    Returns:
        Respective hash value in the format ``$algorithm$salt$hash``.
    """

    salt = _random_string(8)
    h = _pbkdf2(password, salt, algo)
    return '$'.join(['', algo, salt, base64.urlsafe_b64encode(h).decode('utf8')])


def check(password: str, encoded: str) -> bool:
    """Check if a password matches a hash.

    Args:
         password: Password as a string.
         encoded: Hash of the input password as a string.

    Returns:
        ``True`` if password matches the hash, else ``False``.
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
