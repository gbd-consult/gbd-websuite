"""Tests for the mime module."""
import base64
import hmac

import gws
import gws.test.util as u
import gws.lib.password as password
import unittest.mock

# Assuming Salt is random
salt = 'hashtest'

# known_hv from https://www.dcode.fr/pbkdf2-hash with sha512 alg, foo pw, hashtest salt, base 64 output
# known URL safe hashvalue to foo
hvKnown = 'MFyuRkTcnEZd9qB35nslGy/W3a7REHFzpYwYcCTtw1TxRZvqeHwNd0g1DA1DghfkN+OEAcGn32zparlum12UbA=='.replace('/',
                                                                                                             '_').replace(
    '+', '-')


def test_compare_true():
    assert password.compare('foo', 'foo') == True


def test_compare_false():
    assert password.compare('foo', 'bar') == False


def test_encode():
    with unittest.mock.patch('gws.lib.password._random_string', return_value=salt):
        enc = '$'.join(['', 'sha512', salt, hvKnown])
        assert password.encode('foo', 'sha512') == enc


def test_check_true():
    # create encode
    enc = '$'.join(['', 'sha512', salt, hvKnown])
    assert password.check('foo', enc) == True


def test_check_false():
    assert password.check('foo', '$sha512$hashtest$ThisIsJustSomethingDifferent') == False
