import base64

import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    cfg = f'''
        permissions.all "allow all"
        auth.providers+ {{ type "{u.auth.PROVIDER_1}" }}
        auth.methods+ {{ type basic secure False realm "realm_1" }}
        auth.session {{ type "sqlite" }}
        actions [
            {{ type project }}
            {{ type web permissions.read "allow role_1, deny all" }}
        ]
        projects [ {{ uid project_1 permissions.read "allow role_1, deny all" }} ]
    '''

    u.auth.drop_users()
    u.auth.add_user('user_1', 'password_1', roles=['role_1'])
    u.auth.add_user('user_2', 'password_2')

    yield u.gws_root(cfg)


def _header(value):
    return {'Authorization': value}


def _basic(username, password):
    s = base64.b64encode(f'{username}:{password}'.encode('utf8')).decode('ascii')
    return _header(f'Basic {s}')


def _project_info(root, headers=None):
    return u.http.api(root, 'projectInfo', {'projectUid': 'project_1'}, headers=headers or {})


def test_valid_credentials(root: gws.Root):
    res = _project_info(root, _basic('user_1', 'password_1'))
    assert res.status_code == 200


def test_no_header_is_guest(root: gws.Root):
    res = _project_info(root)
    assert res.status_code == 403


def test_wrong_password(root: gws.Root):
    res = _project_info(root, _basic('user_1', 'password_9'))
    assert res.status_code == 403


def test_unknown_user(root: gws.Root):
    res = _project_info(root, _basic('user_9', 'password_1'))
    assert res.status_code == 403


def test_user_without_the_role(root: gws.Root):
    res = _project_info(root, _basic('user_2', 'password_2'))
    assert res.status_code == 403


def test_wrong_scheme(root: gws.Root):
    s = base64.b64encode(b'user_1:password_1').decode('ascii')
    res = _project_info(root, _header(f'Digest {s}'))
    assert res.status_code == 403


def test_header_not_base64(root: gws.Root):
    res = _project_info(root, _header('Basic ???'))
    assert res.status_code == 403


def test_header_without_colon(root: gws.Root):
    s = base64.b64encode(b'user_1').decode('ascii')
    res = _project_info(root, _header(f'Basic {s}'))
    assert res.status_code == 403


def test_empty_username(root: gws.Root):
    s = base64.b64encode(b':password_1').decode('ascii')
    res = _project_info(root, _header(f'Basic {s}'))
    assert res.status_code == 403


def test_get_without_credentials_is_challenged(root: gws.Root):
    res = u.http.get(root, '/_/webAsset?path=file_1')
    assert res.status_code == 401
    assert res.headers['WWW-Authenticate'] == 'Basic realm=realm_1, charset="UTF-8"'


def test_get_with_credentials_is_not_challenged(root: gws.Root):
    res = u.http.get(root, '/_/webAsset?path=file_1', headers=_basic('user_1', 'password_1'))
    assert res.status_code != 401
    assert 'WWW-Authenticate' not in res.headers


def test_post_without_credentials_is_not_challenged(root: gws.Root):
    res = _project_info(root)
    assert res.status_code == 403
    assert 'WWW-Authenticate' not in res.headers


def test_no_session_is_left_behind(root: gws.Root):
    sm = root.app.authMgr.sessionMgr
    sm.delete_all()

    for _ in range(5):
        res = _project_info(root, _basic('user_1', 'password_1'))
        assert res.status_code == 200

    assert len(sm.list_all()) == 0
