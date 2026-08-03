import base64
import urllib.parse

import gws
import gws.lib.otp as otp
import gws.test.util as u


def test_hotp():
    # https://www.rfc-editor.org/rfc/rfc4226#appendix-D

    # The following test data uses the ASCII string
    # "12345678901234567890" for the secret:
    #
    # Table 2 details for each count the truncated values (both in
    # hexadecimal and decimal) and then the HOTP value.
    #
    # Truncated
    # Count    Hexadecimal    Decimal        HOTP
    # 0        4c93cf18       1284755224     755224
    # 1        41397eea       1094287082     287082
    # 2         82fef30        137359152     359152
    # 3        66ef7655       1726969429     969429
    # 4        61c5938a       1640338314     338314
    # 5        33c083d4        868254676     254676
    # 6        7256c032       1918287922     287922
    # 7         4e5b397         82162583     162583
    # 8        2823443f        673399871     399871
    # 9        2679dc69        645520489     520489
    #
    #
    #
    #
    #

    secret = '3132333435363738393031323334353637383930'
    r = [
        '755224',
        '287082',
        '359152',
        '969429',
        '338314',
        '254676',
        '287922',
        '162583',
        '399871',
        '520489',
    ]

    for c in range(10):
        key = bytes.fromhex(secret)
        a = otp.new_hotp(key, c)
        assert a == r.pop(0)


def test_totp():
    # https://www.rfc-editor.org/rfc/rfc6238#appendix-B

    # The test token shared secret uses the ASCII string value
    # "12345678901234567890".  With Time Step X = 30, and the Unix epoch as
    # the initial value to count time steps, where T0 = 0, the TOTP
    # algorithm will display the following values for specified modes and
    #     timestamps.
    #
    # +-------------+--------------+------------------+----------+--------+
    # |  Time (sec) |   UTC Time   | Value of T (hex) |   TOTP   |  Mode  |
    # +-------------+--------------+------------------+----------+--------+
    # |      59     |  1970-01-01  | 0000000000000001 | 94287082 |  SHA1  |
    # |             |   00:00:59   |                  |          |        |
    # |      59     |  1970-01-01  | 0000000000000001 | 46119246 | SHA256 |
    # |             |   00:00:59   |                  |          |        |
    # |      59     |  1970-01-01  | 0000000000000001 | 90693936 | SHA512 |
    # |             |   00:00:59   |                  |          |        |
    # |  1111111109 |  2005-03-18  | 00000000023523EC | 07081804 |  SHA1  |
    # |             |   01:58:29   |                  |          |        |
    # |  1111111109 |  2005-03-18  | 00000000023523EC | 68084774 | SHA256 |
    # |             |   01:58:29   |                  |          |        |
    # |  1111111109 |  2005-03-18  | 00000000023523EC | 25091201 | SHA512 |
    # |             |   01:58:29   |                  |          |        |
    # |  1111111111 |  2005-03-18  | 00000000023523ED | 14050471 |  SHA1  |
    # |             |   01:58:31   |                  |          |        |
    # |  1111111111 |  2005-03-18  | 00000000023523ED | 67062674 | SHA256 |
    # |             |   01:58:31   |                  |          |        |
    # |  1111111111 |  2005-03-18  | 00000000023523ED | 99943326 | SHA512 |
    # |             |   01:58:31   |                  |          |        |
    # |  1234567890 |  2009-02-13  | 000000000273EF07 | 89005924 |  SHA1  |
    # |             |   23:31:30   |                  |          |        |
    # |  1234567890 |  2009-02-13  | 000000000273EF07 | 91819424 | SHA256 |
    # |             |   23:31:30   |                  |          |        |
    # |  1234567890 |  2009-02-13  | 000000000273EF07 | 93441116 | SHA512 |
    # |             |   23:31:30   |                  |          |        |
    # |  2000000000 |  2033-05-18  | 0000000003F940AA | 69279037 |  SHA1  |
    # |             |   03:33:20   |                  |          |        |
    # |  2000000000 |  2033-05-18  | 0000000003F940AA | 90698825 | SHA256 |
    # |             |   03:33:20   |                  |          |        |
    # |  2000000000 |  2033-05-18  | 0000000003F940AA | 38618901 | SHA512 |
    # |             |   03:33:20   |                  |          |        |
    # | 20000000000 |  2603-10-11  | 0000000027BC86AA | 65353130 |  SHA1  |
    # |             |   11:33:20   |                  |          |        |
    # | 20000000000 |  2603-10-11  | 0000000027BC86AA | 77737706 | SHA256 |
    # |             |   11:33:20   |                  |          |        |
    # | 20000000000 |  2603-10-11  | 0000000027BC86AA | 47863826 | SHA512 |
    # |             |   11:33:20   |                  |          |        |
    # +-------------+--------------+------------------+----------+--------+

    seed = "3132333435363738393031323334353637383930"
    seed32 = (
            "3132333435363738393031323334353637383930" +
            "313233343536373839303132")
    seed64 = (
            "3132333435363738393031323334353637383930" +
            "3132333435363738393031323334353637383930" +
            "3132333435363738393031323334353637383930" +
            "31323334")

    r = [
        '94287082',
        '46119246',
        '90693936',
        '07081804',
        '68084774',
        '25091201',
        '14050471',
        '67062674',
        '99943326',
        '89005924',
        '91819424',
        '93441116',
        '69279037',
        '90698825',
        '38618901',
        '65353130',
        '77737706',
        '47863826',
    ]

    for ts in [59, 1111111109, 1111111111, 1234567890, 2000000000, 20000000000]:
        key = bytes.fromhex(seed)
        opts = otp.Options(start=0, step=30, length=8, algo='sha1')
        a = otp.new_totp(key, ts, opts)
        assert a == r.pop(0)

        key = bytes.fromhex(seed32)
        opts = otp.Options(start=0, step=30, length=8, algo='sha256')
        a = otp.new_totp(key, ts, opts)
        assert a == r.pop(0)

        key = bytes.fromhex(seed64)
        opts = otp.Options(start=0, step=30, length=8, algo='sha512')
        a = otp.new_totp(key, ts, opts)
        assert a == r.pop(0)


def test_check_totp_in_tolerance_window():
    secret_1 = 'secret_1'
    ts = 1234567890

    for window in [-1, 0, 1]:
        assert otp.check_totp(otp.new_totp(secret_1, ts + 30 * window), secret_1, ts) is True

    for window in [-2, 2, 100]:
        assert otp.check_totp(otp.new_totp(secret_1, ts + 30 * window), secret_1, ts) is False


def test_check_totp_zero_tolerance():
    secret_1 = 'secret_1'
    ts = 1234567890
    opts = otp.Options(tolerance=0)

    assert otp.check_totp(otp.new_totp(secret_1, ts, opts), secret_1, ts, opts) is True
    assert otp.check_totp(otp.new_totp(secret_1, ts - 30, opts), secret_1, ts, opts) is False


def test_check_totp_wrong_secret():
    secret_1 = 'secret_1'
    secret_2 = 'secret_2'
    ts = 1234567890

    assert otp.check_totp(otp.new_totp(secret_2, ts), secret_1, ts) is False


def test_check_totp_malformed_input():
    secret_1 = 'secret_1'
    ts = 1234567890

    for input in ['', '1', '12345', '1234567', 'abcdef', 'äöüßµ§']:
        assert otp.check_totp(input, secret_1, ts) is False


def test_totp_key_uri():
    secret_1 = 'secret_1'

    scheme, method, label, params = _parse_key_uri(
        otp.totp_key_uri(secret_1, 'Issuer 1', 'user_1@example.com'))

    assert scheme == 'otpauth'
    assert method == 'totp'
    assert label == '/Issuer 1:user_1@example.com'
    assert otp.base32_decode(params['secret']) == b'secret_1'
    assert params['issuer'] == 'Issuer 1'
    assert 'counter' not in params
    assert 'period' not in params
    assert 'digits' not in params
    assert 'algorithm' not in params


def test_totp_key_uri_with_options():
    secret_1 = 'secret_1'
    opts = otp.Options(step=60, length=8, algo='sha256')

    _, method, _, params = _parse_key_uri(
        otp.totp_key_uri(secret_1, 'Issuer 1', 'user_1@example.com', opts))

    assert method == 'totp'
    assert params['period'] == '60'
    assert params['digits'] == '8'
    assert params['algorithm'] == 'sha256'
    assert 'counter' not in params


def test_hotp_key_uri():
    secret_1 = 'secret_1'

    _, method, _, params = _parse_key_uri(
        otp.hotp_key_uri(secret_1, 'Issuer 1', 'user_1@example.com', 42))

    assert method == 'hotp'
    assert params['counter'] == '42'
    assert 'period' not in params


def test_hotp_key_uri_ignores_period():
    secret_1 = 'secret_1'
    opts = otp.Options(step=60)

    _, _, _, params = _parse_key_uri(
        otp.hotp_key_uri(secret_1, 'Issuer 1', 'user_1@example.com', 0, opts))

    assert params['counter'] == '0'
    assert 'period' not in params


def test_base32():
    assert otp.base32_encode('secret_1') == base64.b32encode(b'secret_1').decode('ascii')
    assert otp.base32_encode(b'secret_1') == otp.base32_encode('secret_1')
    assert otp.base32_decode(otp.base32_encode('secret_1')) == b'secret_1'


def test_random_secret():
    for length in [8, 16, 32, 64]:
        assert len(otp.base32_encode(otp.random_secret(length))) == length

    assert otp.random_secret() != otp.random_secret()

    with u.raises(ValueError):
        otp.random_secret(30)


def _parse_key_uri(uri):
    p = urllib.parse.urlsplit(uri)
    return p.scheme, p.netloc, urllib.parse.unquote(p.path), dict(urllib.parse.parse_qsl(p.query))
