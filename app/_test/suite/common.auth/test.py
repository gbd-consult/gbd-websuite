import time

import _test.util as u

BASE = '_/cmd/assetHttpGetPath/path/x.html/projectUid/'
PASS = '123'


def test_noauth():
    assert u.req(BASE + 'public').status_code == 200
    assert u.req(BASE + 'testrole_only').status_code == 403
    assert u.req(BASE + 'user_only').status_code == 403
    assert u.req(BASE + 'guest_only').status_code == 200


def test_login_our_group():
    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASS})
    assert r.status_code == 200

    cookies = r.cookies

    assert u.req(BASE + 'public', cookies=cookies).status_code == 200
    assert u.req(BASE + 'testrole_only', cookies=cookies).status_code == 200
    assert u.req(BASE + 'user_only', cookies=cookies).status_code == 200
    assert u.req(BASE + 'guest_only', cookies=cookies).status_code == 403


def test_login_our_and_other_group():
    r = u.cmd('authLogin', {'username': 'bbb-testrole', 'password': PASS})
    assert r.status_code == 200

    cookies = r.cookies

    assert u.req(BASE + 'public', cookies=cookies).status_code == 200
    assert u.req(BASE + 'testrole_only', cookies=cookies).status_code == 200
    assert u.req(BASE + 'user_only', cookies=cookies).status_code == 200
    assert u.req(BASE + 'guest_only', cookies=cookies).status_code == 403


def test_login_other_group():
    r = u.cmd('authLogin', {'username': 'ccc-no-testrole', 'password': PASS})
    assert r.status_code == 200

    cookies = r.cookies

    assert u.req(BASE + 'public', cookies=cookies).status_code == 200
    assert u.req(BASE + 'testrole_only', cookies=cookies).status_code == 403
    assert u.req(BASE + 'user_only', cookies=cookies).status_code == 200
    assert u.req(BASE + 'guest_only', cookies=cookies).status_code == 403


def test_wrong_login():
    r = u.cmd('authLogin', {'username': 'WRONG', 'password': PASS})
    assert r.status_code == 403


def test_wrong_password():
    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': 'WRONG'})
    assert r.status_code == 403


def test_login_with_wrong_method():
    r = u.cmd('authLogin', {'username': 'basic-login', 'password': PASS})
    assert r.status_code == 403


def test_login_when_logged_in():
    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASS})
    assert r.status_code == 200

    cookies = r.cookies

    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASS}, cookies=cookies)
    assert r.status_code == 403


def test_session_expired():
    assert u.req(BASE + 'testrole_only').status_code == 403

    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASS})
    cookies = r.cookies

    assert u.req(BASE + 'testrole_only', cookies=cookies).status_code == 200

    time.sleep(2)

    assert u.req(BASE + 'testrole_only', cookies=cookies).status_code == 403


def test_session_bad_cookie():
    assert u.req(BASE + 'testrole_only').status_code == 403

    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASS})
    cookies = r.cookies

    assert u.req(BASE + 'testrole_only', cookies=cookies).status_code == 200

    cookies = {'auth': 'WRONG'}

    assert u.req(BASE + 'testrole_only', cookies=cookies).status_code == 403


def test_basic_auth():
    auth = ('basic-login', PASS)

    assert u.req(BASE + 'public', auth=auth).status_code == 200
    assert u.req(BASE + 'testrole_only', auth=auth).status_code == 200
    assert u.req(BASE + 'user_only', auth=auth).status_code == 200
    assert u.req(BASE + 'guest_only', auth=auth).status_code == 403


def test_basic_auth_wrong_login():
    auth = ('WRONG', PASS)

    assert u.req(BASE + 'public', auth=auth).status_code == 403
    assert u.req(BASE + 'testrole_only', auth=auth).status_code == 403
    assert u.req(BASE + 'user_only', auth=auth).status_code == 403
    assert u.req(BASE + 'guest_only', auth=auth).status_code == 403


def test_basic_auth_wrong_password():
    auth = ('basic-login', 'xyz')

    assert u.req(BASE + 'public', auth=auth).status_code == 403
    assert u.req(BASE + 'testrole_only', auth=auth).status_code == 403
    assert u.req(BASE + 'user_only', auth=auth).status_code == 403
    assert u.req(BASE + 'guest_only', auth=auth).status_code == 403
