import _test.util as u

BASE = '_/cmd/assetHttpGetPath/path/x.html/projectUid/'

def _access(cookies=None):
    return [
        u.req(BASE + 'user_project', cookies=cookies).status_code,
        u.req(BASE + 'uid_project', cookies=cookies).status_code,
        u.req(BASE + 'group_project', cookies=cookies).status_code,
        u.req(BASE + 'or_project', cookies=cookies).status_code,
    ]

def test_no_auth():
    assert _access() == [403, 403, 403, 403]


def test_simple_filter():
    r = u.cmd('authLogin', {'username': 'fry', 'password': 'fry'})
    assert r.status_code == 200
    assert _access(r.cookies) == [200, 200, 403, 403]


def test_group_filter():
    r = u.cmd('authLogin', {'username': 'hermes', 'password': 'hermes'})
    assert r.status_code == 200
    assert _access(r.cookies) == [200, 403, 200, 403]


def test_or_filter():
    r = u.cmd('authLogin', {'username': 'leela', 'password': 'leela'})
    assert r.status_code == 200
    assert _access(r.cookies) == [200, 403, 403, 200]

    r = u.cmd('authLogin', {'username': 'bender', 'password': 'bender'})
    assert r.status_code == 200
    assert _access(r.cookies) == [200, 403, 403, 200]


def test_wrong_login():
    r = u.cmd('authLogin', {'username': 'xyz', 'password': 'xyz'})
    assert r.status_code == 403

    r = u.cmd('authLogin', {'username': 'fry', 'password': 'xyz'})
    assert r.status_code == 403

    r = u.cmd('authLogin', {'username': 'fry', 'password': ''})
    assert r.status_code == 403
