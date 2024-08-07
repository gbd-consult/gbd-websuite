"""Test web server.

This server runs in a dedicated docker container during testing
and acts as a mock for our http-related functionality.

This server does almost nothing by default, but the client can "extend" it by providing "snippets".
A snippet is a Python code fragment, which is injected directly into the request handler.

The properties of the request handler (like ``path``) are available as variables in snippets.

With ``return end(content, status, **headers)`` the snippet can return an HTTP response to the client.

When a request arrives, all snippets added so far are executed until one of them returns.

The server understands the following POST requests:

- ``/__add`` reads a snippet from the request body and adds it to the request handler
- ``/__del`` removes all snippets so far
- ``/__set`` removes all and add this one

IT IS AN EXTREMELY BAD IDEA TO RUN THIS SERVER OUTSIDE OF A TEST ENVIRONMENT.

Example of use::

    # set up a snippet

    requests.post('http://mock-server/__add', data=r'''
        if path == '/say-hello' and query.get('x') == 'y':
            return end('HELLO')
    ''')

    # invoke it

    res = requests.get('http://mock-server/say-hello?x=y')
    assert res.text == 'HELLO'

The mockserver runs in a GWS container, so all gws modules are available for import.

"""

import sys
import http.server
import signal
import json
import urllib.parse

import gws

_SNIPPETS = []


class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    body: bytes
    """Raw request body."""
    json: dict
    """Request body decoded as json."""
    method: str
    """GET, POST etc."""
    path: str
    """Url path part."""
    protocol_version = 'HTTP/1.1'
    """Protocol version."""
    query2: dict
    """Query string as a key => [values] dict, e.g. ``{'a': ['1', '2'], ...etc}`` """
    query: dict
    """Query string as a key => value dict, e.g. ``{'a': '1', 'b': '2', ...etc}`` """
    remote_host: str
    """Remote host."""
    remote_port: int
    """Remote post."""
    text: str
    """Request body decoded as utf8."""

    def handle_one_request(self):
        try:
            return super().handle_one_request()
        except Exception as exc:
            _writeln(f'[mockserver] SERVER ERROR: {exc!r}')

    def do_GET(self):
        self.method = 'GET'
        self.prepare(b'')
        if self.path == '/':
            return self.end('OK')
        return self.run_snippets()

    def do_POST(self):
        self.method = 'POST'
        content_length = int(self.headers['Content-Length'])
        self.prepare(self.rfile.read(content_length))

        if self.path == '/__add':
            _SNIPPETS.insert(0, _dedent(self.text))
            return self.end('ok')
        if self.path == '/__del':
            _SNIPPETS[::] = []
            return self.end('ok')
        if self.path == '/__set':
            _SNIPPETS[::] = []
            _SNIPPETS.insert(0, _dedent(self.text))
            return self.end('ok')

        return self.run_snippets()

    def prepare(self, body: bytes):
        self.body = body
        try:
            self.text = self.body.decode('utf8')
        except:
            self.text = ''
        try:
            self.json = json.loads(self.text)
        except:
            self.json = {}

        path, _, qs = self.path.partition('?')
        self.path = path
        self.query = {}
        self.query2 = {}
        if qs:
            self.query2 = urllib.parse.parse_qs(qs)
            self.query = {k: v[0] for k, v in self.query2.items()}

        self.remote_host, self.remote_port = self.client_address

    def run_snippets(self):
        code = '\n'.join([
            'def F():',
            _indent('\n'.join(_SNIPPETS)),
            _indent('return end("?", 404)'),
            'F()'
        ])
        ctx = {**vars(self), 'end': self.end, 'gws': gws}
        try:
            exec(code, ctx)
        except Exception as exc:
            _writeln(f'[mockserver] SNIPPET ERROR: {exc!r}')
            return self.end('Internal Server Error', 500)

    def end(self, content, status=200, **headers):
        hs = {k.lower(): v for k, v in headers.items()}
        ct = hs.pop('content-type', '')

        if isinstance(content, (list, dict)):
            body = json.dumps(content).encode('utf8')
            ct = ct or 'application/json'
        elif isinstance(content, str):
            body = content.encode('utf8')
            ct = ct or 'text/plain'
        else:
            assert isinstance(content, bytes)
            body = content
            ct = ct or 'application/octet-stream'

        hs['content-type'] = ct
        hs['content-length'] = str(len(body))

        self.send_response(status)

        for k, v in hs.items():
            self.send_header(k, v)
        self.end_headers()

        self.wfile.write(body)


def _dedent(s):
    ls = [p.rstrip() for p in s.split('\n')]
    ind = 1e20

    for ln in ls:
        n = len(ln.lstrip())
        if n > 0:
            ind = min(ind, len(ln) - n)

    return '\n'.join(ln[ind:] for ln in ls)


def _indent(s):
    ind = ' ' * 4
    return '\n'.join(ind + ln for ln in s.split('\n'))


def _writeln(s):
    sys.stdout.write(s + '\n')
    sys.stdout.flush()


def main():
    host = '0.0.0.0'
    port = 80

    httpd = http.server.ThreadingHTTPServer((host, port), HTTPRequestHandler)
    signal.signal(signal.SIGTERM, lambda x, y: httpd.shutdown())
    _writeln(f'[mockserver] started on {host}:{port}')
    httpd.serve_forever()


if __name__ == '__main__':
    main()
