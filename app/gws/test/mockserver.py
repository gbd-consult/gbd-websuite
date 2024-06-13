"""Test web server.

This server runs in a dedicated docker container during testing
and acts as a mock for our http-related functionality.

This server does almost nothing by default, but the client can "extend" it by providing "snippets".
A snippet is a Python code fragment, which is injected directly into the request handler.

The following variables are available to snippets ::

    remote_host - remote host
    remote_port - remote port
    path        - url path
    method      - GET/POST
    body        - raw request body (bytes)
    text        - body decoded as utf8
    json        - body decoded as json
    query       - query string key => value dict, e.g. {'a': '1', 'b': '2'}
    query2      - query string key => [values] dict, e.g. {'a': ['1'], 'b': ['2']}

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
        if path == '/say-hello':
            return end('HELLO')
    ''')

    # invoke it

    res = requests.get('http://mock-server/say-hello')
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


def writeln(s):
    sys.stdout.write(s + '\n')
    sys.stdout.flush()


class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    method: str
    body: bytes
    text: str
    json: dict
    query: dict
    query2: dict
    remote_host: str
    remote_port: int

    def handle_one_request(self):
        try:
            return super().handle_one_request()
        except Exception as exc:
            writeln(f'[mockserver] SERVER ERROR: {exc!r}')

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
            _SNIPPETS.insert(0, dedent(self.text))
            return self.end('ok')
        if self.path == '/__del':
            _SNIPPETS[::] = []
            return self.end('ok')
        if self.path == '/__set':
            _SNIPPETS[::] = []
            _SNIPPETS.insert(0, dedent(self.text))
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
            indent('\n'.join(_SNIPPETS)),
            indent('return end("?", 404)'),
            'F()'
        ])
        ctx = {**vars(self), 'end': self.end, 'gws': gws}
        try:
            exec(code, ctx)
        except Exception as exc:
            writeln(f'[mockserver] SNIPPET ERROR: {exc!r}')

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


def dedent(s):
    ls = [p.rstrip() for p in s.split('\n')]
    ind = 1e20

    for ln in ls:
        n = len(ln.lstrip())
        if n > 0:
            ind = min(ind, len(ln) - n)

    return '\n'.join(ln[ind:] for ln in ls)


def indent(s):
    ind = ' ' * 4
    return '\n'.join(ind + ln for ln in s.split('\n'))


def main():
    host = '0.0.0.0'
    port = 80

    httpd = http.server.ThreadingHTTPServer((host, port), HTTPRequestHandler)
    signal.signal(signal.SIGTERM, httpd.shutdown)
    writeln(f'[mockserver] started on {host}:{port}')
    httpd.serve_forever()


if __name__ == '__main__':
    main()
