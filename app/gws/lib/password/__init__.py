"""Password hashing, checking and generation.

Passwords are hashed with PBKDF2 (100000 iterations) and a random salt. The encoded hash
has the form ``$algorithm$salt$hash``, where the hash is base64 (URL-safe) encoded.
``check`` reads the algorithm and salt back from the encoded value.

``generate`` creates random passwords with a configurable length and number of lowercase,
uppercase, digit and punctuation characters. ``generate_with_groups`` does the same for
arbitrary character groups (``SymbolGroup``).

Example::

    import gws.lib.password

    encoded = gws.lib.password.encode('secret')
    gws.lib.password.check('secret', encoded)  # True

    pw = gws.lib.password.generate(min_len=12, max_len=16, min_digit=2)
"""

import base64
import hashlib
import hmac
import random
import string


def compare(a: str, b: str) -> bool:
    """Compare two strings in constant time, to prevent timing attacks.

    Args:
        a: First string.
        b: Second string.

    Returns:
        ``True`` if the strings are equal, ``False`` otherwise.
    """

    return hmac.compare_digest(a.encode('utf8'), b.encode('utf8'))


def encode(password: str, algo: str = 'sha512') -> str:
    """Encode a password into a salted PBKDF2 hash.

    Args:
        password: Plain text password.
        algo: Hash algorithm name, as in ``hashlib``.

    Returns:
        The encoded hash in the format ``$algorithm$salt$hash``.
    """

    salt = _random_string(8)
    h = _pbkdf2(password, salt, algo)
    return '$'.join(['', algo, salt, base64.urlsafe_b64encode(h).decode('utf8')])


def check(password: str, encoded: str) -> bool:
    """Check if a password matches an encoded hash.

    Args:
        password: Plain text password.
        encoded: Encoded hash, as returned by ``encode``.

    Returns:
        ``True`` if the password matches, ``False`` if it does not or if the encoded hash is invalid.
    """

    try:
        _, algo, salt, hs = str(encoded).split('$')
        h1 = base64.urlsafe_b64decode(hs)
        h2 = _pbkdf2(password, salt, algo)
    except (TypeError, ValueError):
        return False

    return hmac.compare_digest(h1, h2)


class SymbolGroup:
    """A group of characters with the minimum and maximum number of occurrences in a generated password."""

    def __init__(self, s, min_len, max_len):
        """Create a symbol group.

        Args:
            s: Characters of the group.
            min_len: Minimum number of characters from this group.
            max_len: Maximum number of characters from this group.
        """
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
    """Generate a random password.

    Args:
        min_len: Minimum password length.
        max_len: Maximum password length.
        min_lower: Minimum number of lowercase letters.
        max_lower: Maximum number of lowercase letters.
        min_upper: Minimum number of uppercase letters.
        max_upper: Maximum number of uppercase letters.
        min_digit: Minimum number of digits.
        max_digit: Maximum number of digits.
        min_punct: Minimum number of punctuation characters.
        max_punct: Maximum number of punctuation characters.

    Returns:
        The password.

    Raises:
        ``ValueError``: If the constraints cannot be satisfied.
    """

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
    """Generate a random password from a list of symbol groups.

    The ``count`` attribute of each group is updated.

    Args:
        groups: Symbol groups.
        min_len: Minimum password length.
        max_len: Maximum password length.

    Returns:
        The password.

    Raises:
        ``ValueError``: If the constraints cannot be satisfied.
    """

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
