"""Test the test utilities."""

import gws.lib.net
import gws.test.util as u


def test_mockserver():
    u.mockserver.set(r'''
        if path == '/say-hello':
            return end('HELLO')
    ''')
    r = gws.lib.net.http_request(u.mockserver.url('/say-hello'))
    assert r.text == 'HELLO'


def test_mockserver_error():
    u.mockserver.set(r'''
        if path == '/say-hello':
            ERR
    ''')
    r = gws.lib.net.http_request(u.mockserver.url('/say-hello'))
    assert r.status_code == 500
