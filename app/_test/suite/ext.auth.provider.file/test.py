import _test.util as u

BASE = '_/cmd/assetHttpGetPath/path/x.html/projectUid/'
PASS = '123'


def test_login():
    assert u.req(BASE + 'testrole_only').status_code == 403

    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASS})
    cookies = r.cookies

    assert u.req(BASE + 'testrole_only', cookies=cookies).status_code == 200


def test_wrong_login():
    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': 'xyz'})
    assert r.status_code == 403

    r = u.cmd('authLogin', {'username': '...', 'password': 'xyz'})
    assert r.status_code == 403
