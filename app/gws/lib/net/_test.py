import gws.lib.net
import gws.lib.test as test


def test_parse_url():
    url = 'http://foo.bar:1234/path/to/file.ext?lower=AA%2FBB%3ACC&UPPER=DDD#hash'
    p = gws.lib.net.parse_url(url)
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
    p = gws.lib.net.make_url(r)
    assert p == 'http://USER:PASS@foo.bar:1234/path/to/file.ext?p1=AA%20A&p2=BB%26B#hash'


def test_request_ok():
    test.mockserv.poke('ok', {'text': 'hello'})
    res = gws.lib.net.http_request(test.mockserv.url('ok'))
    assert (res.ok, res.status_code, res.text) == (True, 200, 'hello')


def test_request_redirect_ok():
    test.mockserv.poke('redirect', {'status_code': 301, 'headers': {'location': test.mockserv.url('ok')}})
    res = gws.lib.net.http_request(test.mockserv.url('redirect'))
    assert (res.ok, res.status_code, res.text) == (True, 200, 'hello')


def test_request_404():
    res = gws.lib.net.http_request(test.mockserv.url('NOT_FOUND'))
    assert (res.ok, res.status_code) == (False, 404)


def test_request_timeout():
    test.mockserv.poke('timeout', {'delay_time': 2})
    res = gws.lib.net.http_request(test.mockserv.url('timeout'), timeout=1)
    assert (res.ok, res.status_code) == (False, 0)
    res = gws.lib.net.http_request(test.mockserv.url('timeout'), timeout=5)
    assert (res.ok, res.status_code) == (True, 200)


def test_request_connection_error():
    res = gws.lib.net.http_request('255.255.255.255')
    assert (res.ok, res.status_code) == (False, 0)


def test_request_valid_response_cached():
    test.mockserv.poke('ok', {'text': 'ORIGINAL'})
    res = gws.lib.net.http_request(test.mockserv.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    test.mockserv.poke('ok', {'text': 'UPDATED'})

    res = gws.lib.net.http_request(test.mockserv.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    res = gws.lib.net.http_request(test.mockserv.url('ok'))
    assert res.text == 'UPDATED'


def test_request_cache_expiration():
    test.mockserv.poke('ok', {'text': 'ORIGINAL'})
    res = gws.lib.net.http_request(test.mockserv.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    test.mockserv.poke('ok', {'text': 'UPDATED'})

    res = gws.lib.net.http_request(test.mockserv.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    test.sleep(4)

    res = gws.lib.net.http_request(test.mockserv.url('ok'), max_age=3)
    assert res.text == 'UPDATED'


def test_request_invalid_response_not_cached():
    test.mockserv.poke('bad', {'status_code': 400, 'text': 'ORIGINAL'})
    res = gws.lib.net.http_request(test.mockserv.url('bad'), max_age=10)
    assert res.text == 'ORIGINAL'

    test.mockserv.poke('bad', {'status_code': 400, 'text': 'UPDATED'})
    res = gws.lib.net.http_request(test.mockserv.url('ok'), max_age=10)
    assert res.text == 'UPDATED'
