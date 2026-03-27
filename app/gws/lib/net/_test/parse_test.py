import gws
import gws.lib.net as net
import gws.lib.osx as osx
import gws.test.util as u


def test_parse_url():
    url = 'http://foo.bar:1234/path/to/file.ext?lower=AA%2FBB%3ACC&UPPER=DDD#hash'
    pp = net.parse_url(url)
    assert isinstance(pp, net.Url)
    assert pp.url == url
    assert vars(pp.pathparts) == vars(osx.ParsePathResult(
        path='/path/to/file.ext',
        dirname='/path/to',
        filename='file.ext',
        stem='file',
        extension='ext',
    ))
    assert pp.params == {'lower': 'AA/BB:CC', 'upper': 'DDD'}
    assert pp.qsl == [('lower', 'AA/BB:CC'), ('UPPER', 'DDD')]
    assert pp.fragment == 'hash'
    assert pp.hostname == 'foo.bar'
    assert pp.netloc == 'foo.bar:1234'
    assert pp.password == ''
    assert pp.port == '1234'
    assert pp.query == 'lower=AA%2FBB%3ACC&UPPER=DDD'
    assert pp.scheme == 'http'
    assert pp.username == ''
    assert pp.password == ''


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


def test_parse_qs():
    qs = 'a=1&b=2&c=3'
    result = net.parse_qs(qs)
    assert result == {'a': ['1'], 'b': ['2'], 'c': ['3']}

    qs = 'a=1&a=2&b=3'
    result = net.parse_qs(qs)
    assert result == {'a': ['1', '2'], 'b': ['3']}

    qs = 'key=value%20with%20spaces&other=test'
    result = net.parse_qs(qs)
    assert result == {'key': ['value with spaces'], 'other': ['test']}


def test_make_qs():
    params = {'a': '1', 'b': '2'}
    result = net.make_qs(params)
    assert result == 'a=1&b=2' or result == 'b=2&a=1'

    params = {'key': 'value with spaces', 'other': 'test&data'}
    result = net.make_qs(params)
    assert 'key=value%20with%20spaces' in result
    assert 'other=test%26data' in result

    # Test with list values
    params = {'colors': ['red', 'green', 'blue']}
    result = net.make_qs(params)
    assert result == 'colors=red%2Cgreen%2Cblue'

    # Test with boolean values
    params = {'active': True, 'deleted': False}
    result = net.make_qs(params)
    assert 'active=true' in result
    assert 'deleted=false' in result

    # Test with list of tuples
    params = [('a', '1'), ('b', '2')]
    result = net.make_qs(params)
    assert result == 'a=1&b=2'


def test_quote_param():
    assert net.quote_param('hello') == 'hello'
    assert net.quote_param('hello world') == 'hello%20world'
    assert net.quote_param('a=b&c=d') == 'a%3Db%26c%3Dd'
    assert net.quote_param('user@host') == 'user%40host'
    assert net.quote_param('path/to/file') == 'path%2Fto%2Ffile'


def test_quote_path():
    assert net.quote_path('path/to/file') == 'path/to/file'
    assert net.quote_path('path with spaces/to/file') == 'path%20with%20spaces/to/file'
    assert net.quote_path('path?query') == 'path%3Fquery'


def test_unquote():
    assert net.unquote('hello') == 'hello'
    assert net.unquote('hello%20world') == 'hello world'
    assert net.unquote('a%3Db%26c%3Dd') == 'a=b&c=d'
    assert net.unquote('user%40host') == 'user@host'


def test_add_params():
    url = 'http://example.com/path'
    result = net.add_params(url, {'key': 'value'})
    assert result == 'http://example.com/path?key=value'

    url = 'http://example.com/path?existing=param'
    result = net.add_params(url, {'new': 'value'})
    assert 'existing=param' in result
    assert 'new=value' in result

    # Test with kwargs
    url = 'http://example.com/path'
    result = net.add_params(url, key1='value1', key2='value2')
    assert 'key1=value1' in result
    assert 'key2=value2' in result


def test_extract_params():
    url = 'http://example.com/path?key1=value1&key2=value2#hash'
    base_url, params = net.extract_params(url)
    assert 'key1=value1' not in base_url
    assert 'key2=value2' not in base_url
    assert params == {'key1': 'value1', 'key2': 'value2'}
    assert '#hash' in base_url


def test_is_abs_url():
    assert net.is_abs_url('http://example.com')
    assert net.is_abs_url('https://example.com')
    assert net.is_abs_url('ftp://example.com')
    assert net.is_abs_url('//example.com')
    assert not net.is_abs_url('example.com')
    assert not net.is_abs_url('/path/to/file')
    assert not net.is_abs_url('relative/path')

