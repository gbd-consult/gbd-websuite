"""Test the test utilities."""

import gws.lib.net
import gws.test.util as u


def test_mockserver():
    u.mockserver.set(r'''
        if path == '/say-hello':
            return end('HELLO=' + query.get('x'))
    ''')
    r = gws.lib.net.http_request(u.mockserver.url('/say-hello?x=y'))
    assert r.text == 'HELLO=y'


def test_mockserver_error():
    u.mockserver.set(r'''
        if path == '/say-hello':
            ERR
    ''')
    r = gws.lib.net.http_request(u.mockserver.url('/say-hello'))
    assert r.status_code == 500
