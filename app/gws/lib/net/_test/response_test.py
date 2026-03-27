import gws
import gws.lib.net as net
import gws.test.util as u


def test_response_init_with_text():
    """Test HTTPResponse initialization with text parameter"""
    res = net.HTTPResponse(ok=True, url='http://example.com', text='Hello World', status_code=200)
    assert res.ok is True
    assert res.url == 'http://example.com'
    assert res.status_code == 200
    assert res.text == 'Hello World'
    assert res.content == b'Hello World'
    assert res.content_type == 'text/plain'
    assert res.content_encoding == 'utf8'


def test_response_init_failed():
    """Test HTTPResponse initialization for failed request"""
    res = net.HTTPResponse(ok=False, url='http://example.com', text='Not Found', status_code=404)
    assert res.ok is False
    assert res.status_code == 404
    assert res.text == 'Not Found'


def test_response_text_property_utf8():
    """Test HTTPResponse.text with UTF-8 encoding"""
    u.mockserver.set(rf"""
        if path == '/utf8':
            return end('Héllo Wörld', content_type='text/plain; charset=utf-8')
    """)
    res = net.http_request(u.mockserver.url('utf8'))
    assert res.ok is True
    assert res.text == 'Héllo Wörld'
    assert res.content_type == 'text/plain'
    assert res.content_encoding == 'utf-8'


def test_response_text_property_latin1():
    """Test HTTPResponse.text with ISO-8859-1 encoding"""
    u.mockserver.set(rf"""
        if path == '/latin1':
            return end('Hello World'.encode('ISO-8859-1'), content_type='text/plain; charset=ISO-8859-1')
    """)
    res = net.http_request(u.mockserver.url('latin1'))
    assert res.ok is True
    assert res.text == 'Hello World'
    assert res.content_encoding == 'ISO-8859-1'


def test_response_text_property_no_charset():
    """Test HTTPResponse.text when no charset is specified (should default to UTF-8)"""
    u.mockserver.set(rf"""
        if path == '/nocharset':
            return end('Hello World', content_type='text/plain')
    """)
    res = net.http_request(u.mockserver.url('nocharset'))
    assert res.ok is True
    assert res.text == 'Hello World'
    assert res.content_type == 'text/plain'
    assert res.content_encoding is None


def test_response_content_type_json():
    """Test HTTPResponse with JSON content type"""
    u.mockserver.set(rf"""
        if path == '/json':
            return end({{'key': 'value'}})
    """)
    res = net.http_request(u.mockserver.url('json'))
    assert res.ok is True
    assert res.content_type == 'application/json'
    assert 'key' in res.text
    assert 'value' in res.text


def test_response_content_type_html():
    """Test HTTPResponse with HTML content type"""
    u.mockserver.set(rf"""
        if path == '/html':
            return end('<html><body>Test</body></html>', content_type='text/html; charset=utf-8')
    """)
    res = net.http_request(u.mockserver.url('html'))
    assert res.ok is True
    assert res.content_type == 'text/html'
    assert '<html>' in res.text


def test_response_content_type_binary():
    """Test HTTPResponse with binary content"""
    u.mockserver.set(rf"""
        if path == '/binary':
            return end(b'\x00\x01\x02\x03', content_type='application/octet-stream')
    """)
    res = net.http_request(u.mockserver.url('binary'))
    assert res.ok is True
    assert res.content_type == 'application/octet-stream'
    assert res.content == b'\x00\x01\x02\x03'


def test_response_raise_if_failed_ok():
    """Test raise_if_failed does not raise for successful response"""
    res = net.HTTPResponse(ok=True, url='http://example.com', text='OK', status_code=200)
    try:
        res.raise_if_failed()
        success = True
    except net.HTTPError:
        success = False
    assert success is True


def test_response_raise_if_failed_error():
    """Test raise_if_failed raises HTTPError for failed response"""
    res = net.HTTPResponse(ok=False, url='http://example.com', text='Not Found', status_code=404)
    try:
        res.raise_if_failed()
        success = False
    except net.HTTPError as e:
        success = True
        assert '404' in str(e.args)
    assert success is True


def test_response_text_caching():
    """Test that HTTPResponse.text is cached after first access"""
    res = net.HTTPResponse(ok=True, url='http://example.com', text='Test', status_code=200)
    text1 = res.text
    text2 = res.text
    assert text1 is text2  # Should be the same object (cached)


def test_response_empty_content():
    """Test HTTPResponse with empty content"""
    res = net.HTTPResponse(ok=True, url='http://example.com', text='', status_code=204)
    assert res.text == ''
    assert res.content == b''


def test_response_none_text():
    """Test HTTPResponse with None text parameter"""
    res = net.HTTPResponse(ok=True, url='http://example.com', text=None, status_code=200)
    assert res.text == ''
    assert res.content == b''
