import gws
import gws.base.auth
import gws.lib.osx as osx
import gws.test.util as u

DB_PATH = u.option('BASE_DIR') + '/throttle'

PROVIDER_2 = 'mockThrottleAuthProvider2'


class Provider2(gws.base.auth.provider.Object):
    """A provider which rejects everything with an error, like the file/ldap/sql providers do."""

    def authenticate(self, method, credentials):
        raise gws.ForbiddenError('wrong password')


def _root(throttle_cfg, provider_type=u.auth.PROVIDER_1):
    osx.unlink(DB_PATH)

    specs = u.gws_specs()
    specs.register_object(gws.ext.object.authProvider, PROVIDER_2, Provider2)

    cfg = f'''
        auth {{
            providers+ {{
                type "{provider_type}"
                allowedMethods [ "{u.auth.METHOD_1}" ]
            }}
            methods+ {{ type "{u.auth.METHOD_1}" }}
            throttle {{
                path {DB_PATH!r}
                {throttle_cfg}
            }}
        }}
    '''

    root = u.gws_root(cfg, specs=specs)

    u.auth.drop_users()
    u.auth.add_user('me', 'foo')
    u.auth.add_user('you', 'bar')

    return root


def _login(root, ip, username='me', password='foo'):
    am = root.app.authMgr
    return am.authenticate(
        am.methods[0],
        gws.Data(username=username, password=password),
        u.auth.requester(root, ip),
    )


##


def test_no_throttle_config_does_not_block():
    osx.unlink(DB_PATH)

    cfg = f'''
        auth {{
            providers+ {{
                type "{u.auth.PROVIDER_1}"
                allowedMethods [ "{u.auth.METHOD_1}" ]
            }}
            methods+ {{ type "{u.auth.METHOD_1}" }}
        }}
    '''
    root = u.gws_root(cfg)
    u.auth.drop_users()
    u.auth.add_user('me', 'foo')

    for _ in range(20):
        assert _login(root, '1.1.1.1', password='WRONG') is None

    assert _login(root, '1.1.1.1') is not None


def test_the_login_command_returns_429():
    osx.unlink(DB_PATH)

    cfg = f'''
        permissions.all "allow all"
        auth.providers+ {{ type "{u.auth.PROVIDER_1}" }}
        auth.methods+ {{ type web secure False }}
        auth.session {{ type "sqlite" }}
        auth.throttle {{ path {DB_PATH!r} maxAttemptsPerIp 2 }}
        actions [ {{ type auth }} ]
    '''
    root = u.gws_root(cfg)
    u.auth.drop_users()
    u.auth.add_user('me', 'foo')

    def _api_login(password):
        return u.http.api(
            root,
            'authLogin',
            {'username': 'me', 'password': password},
            environ_base={'REMOTE_ADDR': '1.1.1.1'},
        )

    assert _api_login('WRONG').status_code == 403
    assert _api_login('WRONG').status_code == 403
    assert _api_login('foo').status_code == 429


def test_blocks_an_address_after_max_attempts():
    root = _root('maxAttemptsPerIp 3')

    for _ in range(3):
        assert _login(root, '1.1.1.1', password='WRONG') is None

    with u.raises(gws.TooManyRequestsError):
        _login(root, '1.1.1.1', password='WRONG')

    # valid credentials are blocked as well
    with u.raises(gws.TooManyRequestsError):
        _login(root, '1.1.1.1')


def test_blocks_an_address_after_max_attempts_with_a_raising_provider():
    root = _root('maxAttemptsPerIp 3', provider_type=PROVIDER_2)

    for _ in range(3):
        with u.raises(gws.ForbiddenError):
            _login(root, '1.1.1.1')

    with u.raises(gws.TooManyRequestsError):
        _login(root, '1.1.1.1')


def test_does_not_block_other_addresses():
    root = _root('maxAttemptsPerIp 3')

    for _ in range(3):
        assert _login(root, '1.1.1.1', password='WRONG') is None

    with u.raises(gws.TooManyRequestsError):
        _login(root, '1.1.1.1')

    assert _login(root, '2.2.2.2') is not None


def test_a_successful_login_resets_the_counter():
    root = _root('maxAttemptsPerIp 3')

    for _ in range(2):
        assert _login(root, '1.1.1.1', password='WRONG') is None

    assert _login(root, '1.1.1.1') is not None

    for _ in range(2):
        assert _login(root, '1.1.1.1', password='WRONG') is None

    assert _login(root, '1.1.1.1') is not None


def test_blocks_a_login_from_any_address():
    root = _root('maxAttemptsPerIp 0 maxAttemptsPerUser 3')

    for ip in ['1.1.1.1', '2.2.2.2', '3.3.3.3']:
        assert _login(root, ip, password='WRONG') is None

    with u.raises(gws.TooManyRequestsError):
        _login(root, '4.4.4.4')

    # other logins are not affected
    assert _login(root, '4.4.4.4', username='you', password='bar') is not None


def test_login_names_are_matched_case_insensitively():
    root = _root('maxAttemptsPerIp 0 maxAttemptsPerUser 3')

    for name in ['me', 'ME', ' Me ']:
        assert _login(root, '1.1.1.1', username=name, password='WRONG') is None

    with u.raises(gws.TooManyRequestsError):
        _login(root, '1.1.1.1')


def test_allow_from_addresses_are_exempt():
    root = _root('maxAttemptsPerIp 3 allowFrom [ "1.1.1.1" ]')

    for _ in range(5):
        assert _login(root, '1.1.1.1', password='WRONG') is None

    assert _login(root, '1.1.1.1') is not None


def test_a_block_expires():
    root = _root('maxAttemptsPerIp 2 windowTime 1 blockTime 2')

    for _ in range(2):
        assert _login(root, '1.1.1.1', password='WRONG') is None

    with u.raises(gws.TooManyRequestsError):
        _login(root, '1.1.1.1')

    gws.u.sleep(3)

    assert _login(root, '1.1.1.1') is not None


def test_block_time_below_window_time_is_an_error():
    with u.raises(gws.ConfigurationError):
        _root('windowTime 10 blockTime 5')


def test_attempts_outside_the_window_are_not_counted():
    root = _root('maxAttemptsPerIp 3 windowTime 1')

    for _ in range(2):
        assert _login(root, '1.1.1.1', password='WRONG') is None

    gws.u.sleep(2)

    for _ in range(2):
        assert _login(root, '1.1.1.1', password='WRONG') is None

    assert _login(root, '1.1.1.1') is not None
