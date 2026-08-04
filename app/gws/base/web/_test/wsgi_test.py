import gws
import gws.base.web.wsgi
import gws.test.util as u


def _root(site_cfg=''):
    cfg = f'''
        web.sites+ {{
            host "*"
            {site_cfg}
        }}
    '''
    return u.gws_root(cfg)


def _request(root, **environ):
    req = gws.base.web.wsgi.Requester(
        root,
        {'REQUEST_METHOD': 'GET', 'PATH_INFO': gws.c.SERVER_ENDPOINT, **environ},
        root.app.webMgr.site,
    )
    req.parse()
    return req


##


def test_ip_is_the_remote_address():
    root = _root()
    req = _request(root, REMOTE_ADDR='1.1.1.1')
    assert req.ip == '1.1.1.1'


def test_forwarded_headers_are_ignored_without_proxy_count():
    root = _root()
    req = _request(
        root,
        REMOTE_ADDR='1.1.1.1',
        HTTP_X_FORWARDED_FOR='9.9.9.9',
        HTTP_X_FORWARDED_PROTO='https',
        HTTP_X_FORWARDED_HOST='example.com',
        HTTP_X_FORWARDED_PORT='8080',
    )
    assert req.ip == '1.1.1.1'
    assert req.scheme == 'http'
    assert req.isSecure is False
    assert req.host == ''
    assert req.port == 0


def test_one_proxy_uses_the_last_forwarded_address():
    root = _root('proxyCount 1')
    req = _request(root, REMOTE_ADDR='1.1.1.1', HTTP_X_FORWARDED_FOR='9.9.9.9, 2.2.2.2')
    assert req.ip == '2.2.2.2'


def test_two_proxies_use_the_second_forwarded_address_from_the_right():
    root = _root('proxyCount 2')
    req = _request(root, REMOTE_ADDR='1.1.1.1', HTTP_X_FORWARDED_FOR='9.9.9.9, 2.2.2.2, 3.3.3.3')
    assert req.ip == '2.2.2.2'


def test_a_short_forwarded_list_falls_back_to_the_remote_address():
    root = _root('proxyCount 3')
    req = _request(root, REMOTE_ADDR='1.1.1.1', HTTP_X_FORWARDED_FOR='9.9.9.9, 2.2.2.2')
    assert req.ip == '1.1.1.1'


def test_a_missing_forwarded_header_falls_back_to_the_remote_address():
    root = _root('proxyCount 1')
    req = _request(root, REMOTE_ADDR='1.1.1.1')
    assert req.ip == '1.1.1.1'


def test_forwarded_host_proto_and_port_are_used_with_proxy_count():
    root = _root('proxyCount 1')
    req = _request(
        root,
        REMOTE_ADDR='1.1.1.1',
        HTTP_HOST='internal.example.com',
        HTTP_X_FORWARDED_HOST='example.com',
        HTTP_X_FORWARDED_PROTO='https',
        HTTP_X_FORWARDED_PORT='8080',
    )
    assert req.host == 'example.com'
    assert req.scheme == 'https'
    assert req.isSecure is True
    assert req.port == 8080


def test_the_forwarded_port_wins_over_the_host_header_port():
    root = _root('proxyCount 1')
    req = _request(
        root,
        HTTP_HOST='internal.example.com:3333',
        HTTP_X_FORWARDED_PORT='8080',
    )
    assert req.port == 8080


def test_the_host_header_port_is_used_without_a_forwarded_port():
    root = _root('proxyCount 1')
    req = _request(root, HTTP_HOST='example.com:1234')
    assert req.port == 1234


def test_an_invalid_forwarded_proto_is_ignored():
    root = _root('proxyCount 1')
    req = _request(root, HTTP_X_FORWARDED_PROTO='ftp')
    assert req.scheme == 'http'


def test_hostnames_are_matched_against_the_forwarded_host():
    root = _root('proxyCount 1 hostnames [ "example.com" ]')

    req = _request(root, HTTP_HOST='internal.example.com', HTTP_X_FORWARDED_HOST='example.com')
    assert req.host == 'example.com'

    with u.raises(gws.base.web.error.BadRequest):
        _request(root, HTTP_HOST='example.com', HTTP_X_FORWARDED_HOST='evil.example.com')
