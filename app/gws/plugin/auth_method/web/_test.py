import gws
import gws.lib.jsonx as jsonx
import gws.test.util as u


@u.fixture(scope='module')
def root():
    cfg = '''
        permissions.all "allow all"

        auth.providers+ {
            type mockAuthProvider1
        }
        auth.mfa+ {
            type mockAuthMfaAdapter1
            uid "MFA_1"
            maxVerifyAttempts 3
        }
        auth.methods+ {
            type web
            secure False
            cookieName AUTH_COOKIE
        }
        auth.session {
            type "sqlite"
        }
        actions [
            { type auth }
            { type project }
        ]
        projects [
            { uid ALL  permissions.read 'allow all' }
            { uid one  permissions.read 'allow role1, deny all' }
            { uid two  permissions.read 'allow role2, deny all' }
        ]
    '''

    yield u.gws_root(cfg)


def _login(root, username, password):
    return u.http.api(root, 'authLogin', {'username': username, 'password': password})


def _get_project(root, project_uid, cookie):
    if cookie is None:
        return u.http.api(root, 'projectInfo', {'projectUid': project_uid})
    else:
        return u.http.api(root, 'projectInfo', {'projectUid': project_uid}, cookies=[cookie])


#


def test_login_ok(root: gws.Root):
    u.mock.add_user('me', 'foo', displayName='123')
    res = _login(root, 'me', 'foo')

    assert res.status_code == 200
    assert res.cookies.get('AUTH_COOKIE') is not None
    assert res.json['user']['displayName'] == '123'


def test_login_wrong_credentials(root: gws.Root):
    u.mock.add_user('me', 'foo', displayName='123')

    assert _login(root, 'XXX', 'foo').status_code == 403
    assert _login(root, '', 'foo').status_code == 403

    assert _login(root, 'me', 'XXX').status_code == 403
    assert _login(root, 'me', '').status_code == 403


def test_request_with_cookie_ok(root: gws.Root):
    u.mock.add_user('one', 'foo', roles=['role1'])

    res = _login(root, 'one', 'foo')
    cookie = res.cookies.get('AUTH_COOKIE')

    assert _get_project(root, 'ALL', cookie).status_code == 200
    assert _get_project(root, 'one', cookie).status_code == 200
    assert _get_project(root, 'two', cookie).status_code == 403

    u.mock.add_user('two', 'bar', roles=['role2'])

    res = _login(root, 'two', 'bar')
    cookie = res.cookies.get('AUTH_COOKIE')

    assert _get_project(root, 'ALL', cookie).status_code == 200
    assert _get_project(root, 'one', cookie).status_code == 403
    assert _get_project(root, 'two', cookie).status_code == 200


def test_request_without_cookie_fails(root: gws.Root):
    u.mock.add_user('one', 'foo', roles=['role1'])
    res = _login(root, 'one', 'foo')
    assert _get_project(root, 'one', None).status_code == 403


def test_request_with_wrong_cookie_fails(root: gws.Root):
    u.mock.add_user('one', 'foo', roles=['role1'])

    res = _login(root, 'one', 'foo')
    cookie = res.cookies.get('AUTH_COOKIE')

    assert _get_project(root, 'one', cookie).status_code == 200
    cookie.value = 'XXX'
    assert _get_project(root, 'one', cookie).status_code == 403


def test_request_with_expired_cookie_fails(root: gws.Root):
    u.mock.add_user('one', 'foo', roles=['role1'])

    ttl = 5
    root.app.authMgr.sessionMgr.lifeTime = ttl

    res = _login(root, 'one', 'foo')
    cookie = res.cookies.get('AUTH_COOKIE')

    gws.u.sleep(ttl - 1)

    res = _get_project(root, 'one', cookie)
    assert res.status_code == 200

    gws.u.sleep(ttl + 1)

    res = _get_project(root, 'one', cookie)
    assert res.status_code == 403


def test_mfa_ok(root: gws.Root):
    u.mock.add_user('one', 'foo', roles=['role1'], mfaUid='MFA_1')

    res = _login(root, 'one', 'foo')
    cookie = res.cookies.get('AUTH_COOKIE')

    # no login yet
    assert _get_project(root, 'one', cookie).status_code == 403

    res = u.http.api(root, 'authMfaVerify', {'payload': {'code': u.mock.AuthMfaAdapter1.VALID_CODE}}, cookies=[cookie])
    assert res.status_code == 200
    cookie = res.cookies.get('AUTH_COOKIE')

    # logged in!
    assert _get_project(root, 'one', cookie).status_code == 200


def test_mfa_retry(root: gws.Root):
    u.mock.add_user('one', 'foo', roles=['role1'], mfaUid='MFA_1')

    res = _login(root, 'one', 'foo')
    cookie = res.cookies.get('AUTH_COOKIE')

    res = u.http.api(root, 'authMfaVerify', {'payload': {'code': 'BAD_1'}}, cookies=[cookie])
    assert res.status_code == 200
    cookie = res.cookies.get('AUTH_COOKIE')
    assert _get_project(root, 'one', cookie).status_code == 403

    res = u.http.api(root, 'authMfaVerify', {'payload': {'code': 'BAD_2'}}, cookies=[cookie])
    assert res.status_code == 200
    cookie = res.cookies.get('AUTH_COOKIE')
    assert _get_project(root, 'one', cookie).status_code == 403

    res = u.http.api(root, 'authMfaVerify', {'payload': {'code': u.mock.AuthMfaAdapter1.VALID_CODE}}, cookies=[cookie])
    assert res.status_code == 200
    cookie = res.cookies.get('AUTH_COOKIE')
    assert _get_project(root, 'one', cookie).status_code == 200


def test_mfa_fail(root: gws.Root):
    u.mock.add_user('one', 'foo', roles=['role1'], mfaUid='MFA_1')

    res = _login(root, 'one', 'foo')
    cookie = res.cookies.get('AUTH_COOKIE')

    res = u.http.api(root, 'authMfaVerify', {'payload': {'code': 'BAD_1'}}, cookies=[cookie])
    assert res.status_code == 200
    cookie = res.cookies.get('AUTH_COOKIE')
    assert _get_project(root, 'one', cookie).status_code == 403

    res = u.http.api(root, 'authMfaVerify', {'payload': {'code': 'BAD_2'}}, cookies=[cookie])
    assert res.status_code == 200
    cookie = res.cookies.get('AUTH_COOKIE')
    assert _get_project(root, 'one', cookie).status_code == 403

    res = u.http.api(root, 'authMfaVerify', {'payload': {'code': 'BAD_3'}}, cookies=[cookie])
    assert res.status_code == 403
