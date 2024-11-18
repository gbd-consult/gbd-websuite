import gws
import gws.lib.net as net
import gws.test.util as u


def test_parse_url():
    url = 'http://foo.bar:1234/path/to/file.ext?lower=AA%2FBB%3ACC&UPPER=DDD#hash'
    p = net.parse_url(url)
    r = {
        'fragment': 'hash',
        'hostname': 'foo.bar',
        'netloc': 'foo.bar:1234',
        'params': {'lower': 'AA/BB:CC', 'upper': 'DDD'},
        'path': '/path/to/file.ext',
        'password': '',
        'pathparts': {
            'dirname': '/path/to',
            'extension': 'ext',
            'filename': 'file.ext',
            'name': 'file'
        },
        'port': '1234',
        'qsl': [('lower', 'AA/BB:CC'), ('UPPER', 'DDD')],
        'query': 'lower=AA%2FBB%3ACC&UPPER=DDD',
        'scheme': 'http',
        'url': url,
        'username': '',
    }
    assert vars(p) == r


def test_make_url():
    r = {
        'fragment': 'hash',
        'hostname': 'foo.bar',
        'params': {'p1': 'AA A', 'p2': 'BB&B'},
        'path': '/path/to/file.ext',
        'password': 'PASS',
        'port': '1234',
        'scheme': 'http',
        'username': 'USER',
    }
    p = net.make_url(r)
    assert p == 'http://USER:PASS@foo.bar:1234/path/to/file.ext?p1=AA%20A&p2=BB%26B#hash'


def test_request_ok():
    u.mockserver.set(rf'''
        if path == '/ok':
            return end('HELLO')
    ''')
    res = net.http_request(u.mockserver.url('ok'))
    assert (res.ok, res.status_code, res.text) == (True, 200, 'HELLO')


def test_request_redirect_ok():
    target_url = u.mockserver.url('ok')
    u.mockserver.set(rf'''
        if path == '/ok':
            return end('HELLO')
        if path == '/redir':
            return end('', status=301, location={target_url!r})
    ''')
    res = net.http_request(u.mockserver.url('redir'))
    assert (res.ok, res.status_code, res.text) == (True, 200, 'HELLO')


def test_request_404():
    res = net.http_request(u.mockserver.url('NOT_FOUND'))
    assert (res.ok, res.status_code) == (False, 404)


def test_request_timeout():
    u.mockserver.set(rf'''
        if path == '/timeout':
            gws.u.sleep(3)
            return end('')
    ''')

    res = net.http_request(u.mockserver.url('timeout'), timeout=1)
    assert (res.ok, res.status_code) == (False, 901)

    res = net.http_request(u.mockserver.url('timeout'), timeout=100)
    assert (res.ok, res.status_code) == (True, 200)


def test_request_connection_error():
    res = net.http_request('255.255.255.255')
    assert (res.ok, res.status_code) == (False, 999)


def test_request_valid_response_cached():
    u.mockserver.set(rf'''
        if path == '/ok':
            return end('ORIGINAL')
    ''')

    res = net.http_request(u.mockserver.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    u.mockserver.set(rf'''
        if path == '/ok':
            return end('UPDATED')
    ''')

    res = net.http_request(u.mockserver.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    res = net.http_request(u.mockserver.url('ok'))
    assert res.text == 'UPDATED'


def test_request_cache_expiration():
    u.mockserver.set(rf'''
        if path == '/ok':
            return end('ORIGINAL')
    ''')
    res = net.http_request(u.mockserver.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    u.mockserver.set(rf'''
        if path == '/ok':
            return end('UPDATED')
    ''')

    res = net.http_request(u.mockserver.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    gws.u.sleep(4)

    res = net.http_request(u.mockserver.url('ok'), max_age=3)
    assert res.text == 'UPDATED'


def test_request_invalid_response_not_cached():
    u.mockserver.set(rf'''
        if path == '/bad':
            return end('ORIGINAL', 400)
    ''')

    res = net.http_request(u.mockserver.url('bad'), max_age=10)
    assert res.text == 'ORIGINAL'

    u.mockserver.set(rf'''
        if path == '/bad':
            return end('UPDATED', 400)
    ''')

    res = net.http_request(u.mockserver.url('bad'), max_age=10)
    assert res.text == 'UPDATED'
