import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    cfg = f'''
        permissions.all "allow all"
        auth.providers+ {{ type "{u.auth.PROVIDER_1}" }}
        auth.methods+ {{ type token secure False header "X-Auth" prefix "Bearer" }}
        auth.session {{ type "sqlite" }}
        actions [ {{ type project }} ]
        projects [ {{ uid project_1 permissions.read "allow role_1, deny all" }} ]
    '''

    u.auth.drop_users()
    u.auth.add_user('user_1', token='token_1', roles=['role_1'])
    u.auth.add_user('user_2', token='token_2')

    yield u.gws_root(cfg)


@u.fixture(scope='module')
def root_no_prefix():
    cfg = f'''
        permissions.all "allow all"
        auth.providers+ {{ type "{u.auth.PROVIDER_1}" }}
        auth.methods+ {{ type token secure False header "X-Auth" }}
        auth.session {{ type "sqlite" }}
        actions [ {{ type project }} ]
        projects [ {{ uid project_1 permissions.read "allow role_1, deny all" }} ]
    '''

    u.auth.drop_users()
    u.auth.add_user('user_1', token='token_1', roles=['role_1'])
    u.auth.add_user('user_2', token='token_2')

    yield u.gws_root(cfg)


def _project_info(root, headers=None):
    return u.http.api(root, 'projectInfo', {'projectUid': 'project_1'}, headers=headers or {})


def test_valid_token(root: gws.Root):
    res = _project_info(root, {'X-Auth': 'Bearer token_1'})
    assert res.status_code == 200


def test_no_header_is_guest(root: gws.Root):
    res = _project_info(root)
    assert res.status_code == 403


def test_unknown_token(root: gws.Root):
    res = _project_info(root, {'X-Auth': 'Bearer token_9'})
    assert res.status_code == 403


def test_user_without_the_role(root: gws.Root):
    res = _project_info(root, {'X-Auth': 'Bearer token_2'})
    assert res.status_code == 403


def test_wrong_prefix(root: gws.Root):
    res = _project_info(root, {'X-Auth': 'Token token_1'})
    assert res.status_code == 403


def test_missing_prefix(root: gws.Root):
    res = _project_info(root, {'X-Auth': 'token_1'})
    assert res.status_code == 403


def test_wrong_header_name(root: gws.Root):
    res = _project_info(root, {'X-Other': 'Bearer token_1'})
    assert res.status_code == 403


def test_without_prefix_configured(root_no_prefix: gws.Root):
    res = _project_info(root_no_prefix, {'X-Auth': 'token_1'})
    assert res.status_code == 200


def test_without_prefix_configured_rejects_two_parts(root_no_prefix: gws.Root):
    res = _project_info(root_no_prefix, {'X-Auth': 'Bearer token_1'})
    assert res.status_code == 403


def test_no_session_is_left_behind(root: gws.Root):
    sm = root.app.authMgr.sessionMgr
    sm.delete_all()

    for _ in range(5):
        res = _project_info(root, {'X-Auth': 'Bearer token_1'})
        assert res.status_code == 200

    assert len(sm.list_all()) == 0
