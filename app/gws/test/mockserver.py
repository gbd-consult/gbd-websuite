"""Test web server.

This server runs in a dedicated docker container during testing
and acts as a mock for our http-related functionality.

This server does almost nothing by default, but the client can "extend" it by providing "snippets".
A snippet is a Python code fragment, which is injected directly into the request handler.
It views the request handler object as ``self`` and must end with a call to ``self.out(text, status)``,
which returns an HTTP response to the client.
When a request arrives, all snippets added so far are executed until one of them returns.

The server understands two POST requests:

- ``/__add`` reads a snippet from the request body and adds it to the request handler
- ``/__del`` removes all snippets so far.

IT IS AN EXTREMELY BAD IDEA TO RUN THIS SERVER OUTSIDE OF A TEST ENVIRONMENT.

Example of use::

    # set up a snippet

    requests.post('http://mock-server/__add', data=r'''
        if self.path == '/say-hello':
            return self.out('HELLO')
    ''')

    # invoke it

    text = requests.get('http://mock-server/say-hello')
    assert text == 'HELLO'

Useful properties of the handler object (``self``) to use in snippets ::

    client_address - tuple[host, port]
    path    - url path
    method  - GET/POST
    body    - raw bytes body>
    text    - body decoded as utf8
    json    - body decoded as json
    query   - query string dict, e.g. {'a': '1', 'b': '2'}
    query2  - query string array dict, e.g. {'a': ['1'], 'b': ['2']}

"""

import http.server
import signal
import json
import sys
import urllib.parse

_SNIPPETS = []


class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    method: str
    body: bytes
    text: str
    json: dict
    query: dict
    query2: dict

    def handle_one_request(self):
        try:
            return super().handle_one_request()
        except Exception as exc:
            print(f'mock: SERVER ERROR: {exc!r}')

    def do_GET(self):
        self.method = 'GET'
        self.prepare(b'')
        if self.path == '/':
            return self.out('OK')
        return self.execute()

    def do_POST(self):
        self.method = 'POST'
        content_length = int(self.headers['Content-Length'])
        self.prepare(self.rfile.read(content_length))

        if self.path == '/__add':
            _SNIPPETS.append(dedent(self.text))
            print(f'mock: snippet added')
            return self.out('ok')
        if self.path == '/__del':
            _SNIPPETS[::] = []
            print(f'mock: snippets removed')
            return self.out('ok')

        return self.execute()

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

    def execute(self):
        try:
            code = 'def HANDLER(self):\n' + indent('\n\n'.join(_SNIPPETS)) + '\n' + indent('return self.out("?", 404)')
            g = {}
            exec(code, g)
            return g['HANDLER'](self)
        except Exception as exc:
            print(f'mock: SNIPPET ERROR: {exc!r}')

    def out(self, content, status=200, content_type=None, headers=None):
        if isinstance(content, (list, dict)):
            content = json.dumps(content).encode('utf8')
            content_type = content_type or 'application/json'
        elif isinstance(content, str):
            content = content.encode('utf8')
            content_type = content_type or 'text/plain'
        else:
            assert isinstance(content, bytes)
            content_type = content_type or 'application/octet-stream'

        h = {
            'content-length': str(len(content)),
            'content-type': content_type,
        }
        if headers:
            for k, v in headers.items():
                h[k.lower()] = v

        self.send_response(status)
        for k, v in h.items():
            self.send_header(k, v)
        self.end_headers()

        self.wfile.write(content)


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
    print(f'mock: started on {host}:{port}')
    httpd.serve_forever()


if __name__ == '__main__':
    main()
