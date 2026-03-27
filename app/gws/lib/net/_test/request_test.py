import gws
import gws.lib.net as net
import gws.test.util as u


def test_request_ok():
    u.mockserver.set(rf"""
        if path == '/ok':
            return end('HELLO')
    """)
    res = net.http_request(u.mockserver.url('ok'))
    assert (res.ok, res.status_code, res.text) == (True, 200, 'HELLO')


def test_request_redirect_ok():
    target_url = u.mockserver.url('ok')
    u.mockserver.set(rf"""
        if path == '/ok':
            return end('HELLO')
        if path == '/redir':
            return end('', status=301, location={target_url!r})
    """)
    res = net.http_request(u.mockserver.url('redir'))
    assert (res.ok, res.status_code, res.text) == (True, 200, 'HELLO')


def test_request_404():
    res = net.http_request(u.mockserver.url('NOT_FOUND'))
    assert (res.ok, res.status_code) == (False, 404)


def test_request_timeout():
    u.mockserver.set(rf"""
        if path == '/timeout':
            gws.u.sleep(3)
            return end('')
    """)

    res = net.http_request(u.mockserver.url('timeout'), timeout=1)
    assert (res.ok, res.status_code) == (False, 901)

    res = net.http_request(u.mockserver.url('timeout'), timeout=100)
    assert (res.ok, res.status_code) == (True, 200)


def test_request_connection_error():
    res = net.http_request('255.255.255.255')
    assert (res.ok, res.status_code) == (False, 999)


def test_request_valid_response_cached():
    u.mockserver.set(rf"""
        if path == '/ok':
            return end('ORIGINAL')
    """)

    res = net.http_request(u.mockserver.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    u.mockserver.set(rf"""
        if path == '/ok':
            return end('UPDATED')
    """)

    res = net.http_request(u.mockserver.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    res = net.http_request(u.mockserver.url('ok'))
    assert res.text == 'UPDATED'


def test_request_cache_expiration():
    u.mockserver.set(rf"""
        if path == '/ok':
            return end('ORIGINAL')
    """)
    res = net.http_request(u.mockserver.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    u.mockserver.set(rf"""
        if path == '/ok':
            return end('UPDATED')
    """)

    res = net.http_request(u.mockserver.url('ok'), max_age=3)
    assert res.text == 'ORIGINAL'

    gws.u.sleep(4)

    res = net.http_request(u.mockserver.url('ok'), max_age=3)
    assert res.text == 'UPDATED'


def test_request_invalid_response_not_cached():
    u.mockserver.set(rf"""
        if path == '/bad':
            return end('ORIGINAL', 400)
    """)

    res = net.http_request(u.mockserver.url('bad'), max_age=10)
    assert res.text == 'ORIGINAL'

    u.mockserver.set(rf"""
        if path == '/bad':
            return end('UPDATED', 400)
    """)

    res = net.http_request(u.mockserver.url('bad'), max_age=10)
    assert res.text == 'UPDATED'


def test_request_post():
    """Test POST request with data"""
    u.mockserver.set(rf"""
        if path == '/post' and method == 'POST':
            return end(f'Received: {{text}}')
    """)
    res = net.http_request(u.mockserver.url('post'), method='POST', data='test data')
    assert res.ok is True
    assert 'Received: test data' in res.text


def test_request_post_json():
    """Test POST request with JSON data"""
    u.mockserver.set(rf"""
        if path == '/post-json' and method == 'POST':
            return end({{'received': json}})
    """)
    data = {'key': 'value', 'number': 42}
    res = net.http_request(u.mockserver.url('post-json'), method='POST', json=data)
    assert res.ok is True
    assert 'key' in res.text
    assert 'value' in res.text


def test_request_custom_headers():
    """Test request with custom headers"""
    u.mockserver.set(rf"""
        if path == '/headers':
            custom_header = headers.get('X-Custom-Header', '')
            return end(f'Header: {{custom_header}}')
    """)
    res = net.http_request(u.mockserver.url('headers'), headers={'X-Custom-Header': 'TestValue'})
    assert res.ok is True
    assert 'Header: TestValue' in res.text


def test_request_user_agent():
    """Test that User-Agent header is set by default"""
    u.mockserver.set(rf"""
        if path == '/ua':
            ua = headers.get('User-Agent', '')
            return end(f'UA: {{ua}}')
    """)
    res = net.http_request(u.mockserver.url('ua'))
    assert res.ok is True
    assert 'GBD WebSuite' in res.text


def test_request_params():
    """Test request with params parameter"""
    u.mockserver.set(rf"""
        if path == '/params':
            p1 = query.get('param1', '')
            p2 = query.get('param2', '')
            return end(f'p1={{p1}}, p2={{p2}}')
    """)
    res = net.http_request(u.mockserver.url('params'), params={'param1': 'value1', 'param2': 'value2'})
    assert res.ok is True
    assert 'p1=value1' in res.text
    assert 'p2=value2' in res.text


def test_request_params_with_special_chars():
    """Test request with params containing special characters"""
    u.mockserver.set(rf"""
        if path == '/params-special':
            p = query.get('param', '')
            return end(f'param={{p}}')
    """)
    res = net.http_request(u.mockserver.url('params-special'), params={'param': 'hello world&test=value'})
    assert res.ok is True
    assert 'param=hello world&test=value' in res.text


def test_request_multiple_methods():
    """Test different HTTP methods on the same endpoint"""
    u.mockserver.set(rf"""
        if path == '/multi':
            return end(f'Method: {{method}}')
    """)
    
    res_get = net.http_request(u.mockserver.url('multi'), method='GET')
    assert 'Method: GET' in res_get.text
    
    res_post = net.http_request(u.mockserver.url('multi'), method='POST')
    assert 'Method: POST' in res_post.text


def test_request_with_query_string():
    """Test request with query string in URL"""
    u.mockserver.set(rf"""
        if path == '/query':
            q = query.get('search', '')
            return end(f'Search: {{q}}')
    """)
    res = net.http_request(u.mockserver.url('query?search=test'))
    assert res.ok is True
    assert 'Search: test' in res.text


def test_request_status_codes():
    """Test various HTTP status codes"""
    # Test 201 Created
    u.mockserver.set(rf"""
        if path == '/created':
            return end('Created', status=201)
    """)
    res = net.http_request(u.mockserver.url('created'))
    assert res.ok is True
    assert res.status_code == 201

    # Test 204 No Content
    u.mockserver.set(rf"""
        if path == '/no-content':
            return end('', status=204)
    """)
    res = net.http_request(u.mockserver.url('no-content'))
    assert res.ok is True
    assert res.status_code == 204

    # Test 400 Bad Request
    u.mockserver.set(rf"""
        if path == '/bad-request':
            return end('Bad Request', status=400)
    """)
    res = net.http_request(u.mockserver.url('bad-request'))
    assert res.ok is False
    assert res.status_code == 400

    # Test 500 Internal Server Error
    u.mockserver.set(rf"""
        if path == '/server-error':
            return end('Server Error', status=500)
    """)
    res = net.http_request(u.mockserver.url('server-error'))
    assert res.ok is False
    assert res.status_code == 500


def test_request_with_fragment():
    """Test request URL with fragment (should be ignored)"""
    u.mockserver.set(rf"""
        if path == '/fragment':
            return end('Success')
    """)
    res = net.http_request(u.mockserver.url('fragment#section'))
    assert res.ok is True
    assert res.text == 'Success'


def test_request_empty_response():
    """Test request that returns empty response"""
    u.mockserver.set(rf"""
        if path == '/empty':
            return end('')
    """)
    res = net.http_request(u.mockserver.url('empty'))
    assert res.ok is True
    assert res.text == ''


def test_request_binary_response():
    """Test request that returns binary data"""
    u.mockserver.set(rf"""
        if path == '/binary':
            return end(b'\x00\x01\x02\x03\x04', content_type='application/octet-stream')
    """)
    res = net.http_request(u.mockserver.url('binary'))
    assert res.ok is True
    assert res.content == b'\x00\x01\x02\x03\x04'
    assert res.content_type == 'application/octet-stream'


def test_request_large_response():
    """Test request that returns a large response"""
    u.mockserver.set(rf"""
        if path == '/large':
            return end('X' * 10000)
    """)
    res = net.http_request(u.mockserver.url('large'))
    assert res.ok is True
    assert len(res.text) == 10000
    assert res.text == 'X' * 10000


def test_request_multiple_redirects():
    """Test request with multiple redirects"""
    target_url = u.mockserver.url('final')
    redir2_url = u.mockserver.url('redir2')
    u.mockserver.set(rf"""
        if path == '/final':
            return end('FINAL')
        if path == '/redir2':
            return end('', status=302, location={target_url!r})
        if path == '/redir1':
            return end('', status=301, location={redir2_url!r})
    """)
    res = net.http_request(u.mockserver.url('redir1'))
    assert res.ok is True
    assert res.text == 'FINAL'


def test_request_case_insensitive_method():
    """Test that method parameter is case-insensitive"""
    u.mockserver.set(rf"""
        if path == '/method-test' and method == 'POST':
            return end('POST OK')
    """)
    res = net.http_request(u.mockserver.url('method-test'), method='post')
    assert res.ok is True
    assert res.text == 'POST OK'

