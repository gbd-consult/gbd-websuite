import _test.util as u

base = '_/cmd/assetHttpGetPath/path/x.html/projectUid/'
password = '123'


def test_noauth():
    assert u.req(base + 'public').status_code == 200
    assert u.req(base + 'testrole_only').status_code == 403
    assert u.req(base + 'user_only').status_code == 403
    assert u.req(base + 'guest_only').status_code == 200


def test_login_our_group():
    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': password})
    assert r.status_code == 200

    cookies = r.cookies

    assert u.req(base + 'public', cookies=cookies).status_code == 200
    assert u.req(base + 'testrole_only', cookies=cookies).status_code == 200
    assert u.req(base + 'user_only', cookies=cookies).status_code == 200
    assert u.req(base + 'guest_only', cookies=cookies).status_code == 403


def test_login_our_and_other_group():
    r = u.cmd('authLogin', {'username': 'bbb-testrole', 'password': password})
    assert r.status_code == 200

    cookies = r.cookies

    assert u.req(base + 'public', cookies=cookies).status_code == 200
    assert u.req(base + 'testrole_only', cookies=cookies).status_code == 200
    assert u.req(base + 'user_only', cookies=cookies).status_code == 200
    assert u.req(base + 'guest_only', cookies=cookies).status_code == 403


def test_login_other_group():
    r = u.cmd('authLogin', {'username': 'ccc-no-testrole', 'password': password})
    assert r.status_code == 200

    cookies = r.cookies

    assert u.req(base + 'public', cookies=cookies).status_code == 200
    assert u.req(base + 'testrole_only', cookies=cookies).status_code == 403
    assert u.req(base + 'user_only', cookies=cookies).status_code == 200
    assert u.req(base + 'guest_only', cookies=cookies).status_code == 403


def test_wrong_login():
    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': 'xyz'})
    assert r.status_code == 403

    r = u.cmd('authLogin', {'username': '...', 'password': 'xyz'})
    assert r.status_code == 403


def test_login_when_logged_in():
    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': password})
    assert r.status_code == 200

    cookies = r.cookies

    r = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': password}, cookies=cookies)
    assert r.status_code == 403
