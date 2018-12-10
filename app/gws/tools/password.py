import base64
import hashlib
import hmac
import random


def cmp(a, b):
    return 1 if hmac.compare_digest(a, b) else 0


def encode(password, algo='sha512'):
    salt = _random_string(8)
    h = _hash(password, salt, algo)
    return '$'.join(['', algo, salt, base64.urlsafe_b64encode(h).decode('utf8')])


def check(password, enc):
    try:
        _, algo, salt, hs = str(enc).split('$')
        h1 = base64.urlsafe_b64decode(hs)
        h2 = _hash(password, salt, algo)
    except (TypeError, ValueError):
        return False

    return cmp(h1, h2)


def _hash(password, salt, algo):
    return hashlib.pbkdf2_hmac(algo, password.encode('utf8'), salt.encode('utf8'), 100000)


def _random_string(length):
    a = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    r = random.SystemRandom()
    return ''.join(r.choice(a) for _ in range(length))
