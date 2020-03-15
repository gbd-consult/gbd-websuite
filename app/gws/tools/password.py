import base64
import hashlib
import hmac
import random


def cmp(a, b):
    return hmac.compare_digest(a, b)


def encode(password, algo='sha512'):
    salt = _random_string(8)
    h = _pbkdf2(password, salt, algo)
    return '$'.join(['', algo, salt, base64.urlsafe_b64encode(h).decode('utf8')])


def check(password, enc):
    try:
        _, algo, salt, hs = str(enc).split('$')
        h1 = base64.urlsafe_b64decode(hs)
        h2 = _pbkdf2(password, salt, algo)
    except (TypeError, ValueError):
        return False

    return cmp(h1, h2)


def as_hash(s, algo='sha512'):
    h = hashlib.new(algo)
    if not isinstance(s, bytes):
        s = str(s).encode('utf8')
    h.update(s)
    return h.hexdigest()


def _pbkdf2(password, salt, algo):
    return hashlib.pbkdf2_hmac(algo, password.encode('utf8'), salt.encode('utf8'), 100000)


def _random_string(length):
    a = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    r = random.SystemRandom()
    return ''.join(r.choice(a) for _ in range(length))
