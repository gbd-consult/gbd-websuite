"""Tests for the password module."""

import gws
import gws.test.util as u
import gws.lib.password as password
import unittest.mock

# Assuming Salt is random
salt = 'hashtest'

# known_hv from https://www.dcode.fr/pbkdf2-hash with sha512 alg, foo pw, hashtest salt, base 64 output
# known URL safe hashvalue to foo
hvKnown = 'MFyuRkTcnEZd9qB35nslGy_W3a7REHFzpYwYcCTtw1TxRZvqeHwNd0g1DA1DghfkN-OEAcGn32zparlum12UbA=='


def test_compare_true():
    assert password.compare('foo', 'foo')


def test_compare_false():
    assert not password.compare('foo', 'bar')


def test_encode():
    with unittest.mock.patch('gws.lib.password._random_string', return_value=salt):
        enc = '$'.join(['', 'sha512', salt, hvKnown])
        assert password.encode('foo', 'sha512') == enc


def test_check_true():
    # create encode
    enc = '$'.join(['', 'sha512', salt, hvKnown])
    assert password.check('foo', enc)


def test_check_false():
    assert not password.check('foo', '$sha512$hashtest$ThisIsJustSomethingDifferent')


import string


def _cc(s, ls):
    return sum(c in ls for c in s)


def test_generate_min():
    for _ in range(50):
        g = password.generate(min_len=20, max_len=40, min_lower=5, min_upper=6, min_digit=7, min_punct=8)
        assert 20 <= len(g) <= 40
        assert _cc(g, string.ascii_lowercase) >= 5
        assert _cc(g, string.ascii_uppercase) >= 6
        assert _cc(g, string.digits) >= 7
        assert _cc(g, string.punctuation) >= 8


def test_generate_max():
    for _ in range(50):
        g = password.generate(min_len=20, max_len=40, max_lower=5, max_upper=6, max_digit=7)
        assert 20 <= len(g) <= 40
        assert _cc(g, string.ascii_lowercase) <= 5
        assert _cc(g, string.ascii_uppercase) <= 6
        assert _cc(g, string.digits) <= 7


def test_generate_zero():
    for _ in range(50):
        g = password.generate(min_len=20, max_len=40, max_lower=0)
        assert _cc(g, string.ascii_lowercase) == 0


def test_generate_exact():
    for _ in range(50):
        g = password.generate(min_len=12, max_len=12, min_lower=3, min_upper=3, min_digit=3, min_punct=3)
        assert len(g) == 12
        assert _cc(g, string.ascii_lowercase) == 3
        assert _cc(g, string.ascii_uppercase) == 3
        assert _cc(g, string.digits) == 3
        assert _cc(g, string.punctuation) == 3


def test_generate_mins_greater_than_len():
    for _ in range(50):
        with u.raises(ValueError):
            password.generate(min_len=10, max_len=10, min_lower=3, min_upper=3, min_digit=3, min_punct=3)


def test_generate_maxes_less_than_len():
    for _ in range(50):
        with u.raises(ValueError):
            password.generate(min_len=20, max_len=20, max_lower=3, max_upper=3, max_digit=3, max_punct=3)
