import gws.lib.net as net
import gws.lib.test as test


def test_parse_url():
    url = 'http://foo.bar:1234/path/to/file.ext?lower=AA,BB,CC&UPPER=DDD#hash'
    p = net.parse_url(url)
    r = {
        'fragment': 'hash',
        'hostname': 'foo.bar',
        'netloc': 'foo.bar:1234',
        'params': {'lower': 'AA,BB,CC', 'upper': 'DDD'},
        'path': '/path/to/file.ext',
        'password': '',
        'pathparts': {
            'dirname': '/path/to',
            'ext': 'ext',
            'filename': 'file.ext',
            'name': 'file'
        },
        'port': '1234',
        'qs': {'lower': ['AA,BB,CC'], 'UPPER': ['DDD']},
        'query': 'lower=AA,BB,CC&UPPER=DDD',
        'scheme': 'http',
        'url': url,
        'username': '',
    }
    assert vars(p) == r


def test_make_url():
    r = {
        'fragment': 'hash',
        'hostname': 'foo.bar',
        'params': {'p1': 'AAA', 'p2': 'BBB'},
        'path': '/path/to/file.ext',
        'password': 'PASS',
        'port': '1234',
        'scheme': 'http',
        'username': 'USER',
    }
    p = net.make_url(r)
    assert p == 'http://USER:PASS@foo.bar:1234/path/to/file.ext?p1=AAA&p2=BBB#hash'

