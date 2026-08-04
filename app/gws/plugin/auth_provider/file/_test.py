import time

import gws
import gws.lib.jsonx
import gws.lib.password
import gws.test.util as u

USERS_PATH = '/tmp/users.json'

MANY = 200


def _write_users(count):
    recs = [
        {'login': 'me', 'password': gws.lib.password.encode('foo'), 'name': 'Me', 'roles': ['role1']},
    ]
    for n in range(count):
        recs.append({'login': f'user_{n}', 'password': gws.lib.password.encode(f'pass_{n}')})
    gws.lib.jsonx.to_path(USERS_PATH, recs)


@u.fixture(scope='module')
def root():
    _write_users(MANY)

    cfg = f'''
        auth {{
            methods+ {{ type "basic" secure False }}
            providers+ {{
                type "file"
                path {USERS_PATH!r}
            }}
        }}
    '''

    yield u.gws_root(cfg)


def _provider(root):
    return root.app.authMgr.providers[0]


def _auth(root, username, password):
    return _provider(root).authenticate(root.app.authMgr.methods[0], gws.Data(username=username, password=password))


##


def test_authenticate_valid_user(root: gws.Root):
    usr = _auth(root, 'me', 'foo')
    assert usr.loginName == 'me'
    assert usr.displayName == 'Me'


def test_authenticate_wrong_password(root: gws.Root):
    with u.raises(gws.ForbiddenError):
        _auth(root, 'me', 'WRONG')


def test_authenticate_unknown_user(root: gws.Root):
    assert _auth(root, 'NOBODY', 'foo') is None


def test_authenticate_missing_credentials(root: gws.Root):
    assert _auth(root, '', 'foo') is None
    assert _auth(root, 'me', '') is None


def test_authenticate_hashes_once_regardless_of_user_count(root: gws.Root):
    # the cost of a login attempt must not scale with the number of records,
    # otherwise a single unauthenticated request burns O(n) * pbkdf2
    t = time.time()
    gws.lib.password.check('foo', gws.lib.password.encode('foo'))
    one_check = time.time() - t

    t = time.time()
    _auth(root, 'NOBODY', 'foo')
    unknown_user = time.time() - t

    assert unknown_user < one_check * 5


def test_authenticate_unknown_user_costs_the_same_as_a_known_one(root: gws.Root):
    t = time.time()
    _auth(root, 'NOBODY', 'foo')
    unknown_user = time.time() - t

    t = time.time()
    with u.raises(gws.ForbiddenError):
        _auth(root, 'me', 'WRONG')
    known_user = time.time() - t

    ratio = max(unknown_user, known_user) / max(min(unknown_user, known_user), 0.000001)
    assert ratio < 5


def test_get_user(root: gws.Root):
    usr = _provider(root).get_user('me')
    assert usr.loginName == 'me'
    assert 'role1' in usr.roles


def test_get_unknown_user(root: gws.Root):
    assert _provider(root).get_user('NOBODY') is None
