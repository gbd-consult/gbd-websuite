"""Test the test utilities."""

import gws.lib.net
import gws.test.util as u


def test_mockserver_str():
    u.mockserver_add_snippet(r'''
        if self.path == '/say-hello':
            return self.out('HELLO')
    ''')
    r = u.mockserver_invoke('/say-hello')
    assert r == 'HELLO'


def test_mockserver_bytes():
    u.mockserver_add_snippet(r'''
        if self.path == '/say-hello-bytes':
            return self.out(b'HELLO-BYTES')
    ''')
    r = u.mockserver_invoke('/say-hello-bytes')
    assert r == b'HELLO-BYTES'


def test_mockserver_json():
    u.mockserver_add_snippet(r'''
        if self.path == '/say-json':
            return self.out({'foo': 'bar'})
    ''')
    r = u.mockserver_invoke('/say-json')
    assert r == {'foo': 'bar'}


def test_mockserver_clear():
    u.mockserver_add_snippet(r'''
        if self.path == '/say-hello':
            return self.out('HELLO')
    ''')
    r = u.mockserver_invoke('/say-hello')
    assert r == 'HELLO'

    u.mockserver_clear()

    r = u.mockserver_invoke('/say-hello')
    assert r is None
