"""Password tools."""

import base64
import hashlib
import hmac
import random
import string


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


class SymbolGroup:
    def __init__(self, s, min_len, max_len):
        self.chars = s
        self.max = max_len
        self.min = min_len
        self.count = 0


def generate(
        min_len: int = 16,
        max_len: int = 16,
        min_lower: int = 0,
        max_lower: int = 255,
        min_upper: int = 0,
        max_upper: int = 255,
        min_digit: int = 0,
        max_digit: int = 255,
        min_punct: int = 0,
        max_punct: int = 255,
) -> str:
    """Generate a random password."""

    groups = [
        SymbolGroup(string.ascii_lowercase, min_lower, max_lower),
        SymbolGroup(string.ascii_uppercase, min_upper, max_upper),
        SymbolGroup(string.digits, min_digit, max_digit),
        SymbolGroup(string.punctuation, min_punct, max_punct),
    ]
    return generate_with_groups(groups, min_len, max_len)


def generate_with_groups(
        groups: list[SymbolGroup],
        min_len: int = 16,
        max_len: int = 16,
) -> str:
    """Generate a random password from a list of `SymbolGroup` objects."""

    r = random.SystemRandom()
    p = []

    for g in groups:
        p.extend(r.choices(g.chars, k=g.min))
        g.count = g.min

    if len(p) > max_len:
        raise ValueError('invalid parameters')

    size = r.randint(max(min_len, len(p)), max_len)

    while len(p) < size:
        sel = ''.join(g.chars for g in groups if g.count < g.max)
        if not sel:
            raise ValueError('invalid parameters')
        c = r.choice(sel)
        for g in groups:
            if c in g.chars:
                g.count += 1
                break
        p.append(c)

    r.shuffle(p)

    return ''.join(p)


##


def _pbkdf2(password, salt, algo):
    return hashlib.pbkdf2_hmac(algo, password.encode('utf8'), salt.encode('utf8'), 100000)


def _random_string(length):
    a = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    r = random.SystemRandom()
    return ''.join(r.choice(a) for _ in range(length))
