import time

import gws
import gws.lib.json2
import gws.lib.test as test

COOKIE_NAME = 'TESTAUTH'
SESSION_LIFETIME = 2


@test.fixture(scope='module', autouse=True)
def configuration():
    test.setup()

    users_json_path = test.make_users_json([
        {
            'login': 'user1',
            'password': '123',
            'roles': ['role1'],
        },
        {
            'login': 'user2',
            'password': '345',
            'roles': ['role2'],
        },
    ])

    test.configure(f'''
        auth.providers+ {{
            type file
            path: {users_json_path!r}
        }}
        
        auth.methods+ {{
            type web
            secure False
            cookieName {COOKIE_NAME!r}
        }}
        
        auth.sessionLifeTime {SESSION_LIFETIME!r}

        api.actions [
            {{ type auth     access+ {{ type allow role all }} }}
            {{ type project  access+ {{ type allow role all }} }}
        ]
        
        projects [
            {{ uid ALL  access+ {{ type allow role all }} }}
            {{ uid one  access+ {{ type allow role role1 }} }}
            {{ uid two  access+ {{ type allow role role2 }} }}
        ]
    ''')

    yield

    test.teardown()


def _login(username, password):
    return gws.lib.test.client_cmd_request('authLogin', {'username': username, 'password': password})


def _access(project_uid, cookie):
    if cookie is None:
        return gws.lib.test.client_cmd_request('projectInfo', {'projectUid': project_uid})
    else:
        return gws.lib.test.client_cmd_request('projectInfo', {'projectUid': project_uid}, cookies={COOKIE_NAME: cookie})


#


def test_login_ok():
    r = _login('user1', '123')
    assert r.status == 200
    assert len(r.cookies[COOKIE_NAME]['value']) > 0
    assert r.json['user']['displayName'] == 'user1'


def test_login_wrong_username():
    assert _login('XXX', '123').status == 403
    assert _login('', '123').status == 403


def test_login_wrong_password():
    assert _login('user1', 'XXX').status == 403
    assert _login('user1', '').status == 403


def test_request_with_cookie_ok():
    r = _login('user1', '123')
    assert r.status == 200
    sid1 = r.cookies[COOKIE_NAME]['value']

    assert _access('ALL', sid1).status == 200
    assert _access('one', sid1).status == 200
    assert _access('two', sid1).status == 403

    r = _login('user2', '345')
    sid2 = r.cookies[COOKIE_NAME]['value']

    assert _access('ALL', sid2).status == 200
    assert _access('one', sid2).status == 403
    assert _access('two', sid2).status == 200


def test_request_without_cookie_fails():
    assert _login('user1', '123').status == 200
    assert _access('one', None).status == 403


def test_request_with_wrong_cookie_fails():
    assert _login('user1', '123').status == 200
    assert _access('one', 'XXX').status == 403
    assert _access('one', '').status == 403


def test_request_with_expired_cookie_fails():
    r = _login('user1', '123')
    sid1 = r.cookies[COOKIE_NAME]['value']

    time.sleep(SESSION_LIFETIME - 1)

    r = _access('one', sid1)
    assert r.status == 200
    assert len(r.cookies[COOKIE_NAME]['value']) > 1

    time.sleep(SESSION_LIFETIME + 2)

    r = _access('one', sid1)
    assert r.status == 403
    assert len(r.cookies[COOKIE_NAME]['value']) == 0
