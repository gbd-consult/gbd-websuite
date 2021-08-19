"""Test web server.

This servers runs in a dedicated docker container during testing
and acts as a mock for our http-related functonality.
"""

import http.server
import json
import re
import os

# from PIL import Image, ImageColor, ImageDraw, ImageFont


STATE = {
    'cdir': os.path.dirname(__file__),
    'responses': {},
    'capture': None,
}


class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path

        if STATE['capture'] is not None:
            STATE['capture'].append(path)

        for k, v in STATE['responses'].items():
            if v and re.search(k, path):
                return self.end(200, v)

        file = STATE['cdir'] + '/' + path
        if os.path.isfile(file):
            with open(file, 'rb') as fp:
                self.end(200, fp.read())

        self.end(404, 'not found\n')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post = json.loads(self.rfile.read(content_length))
        res = getattr(self, 'cmd_' + post['cmd'])(post)
        if not res:
            res = {'ok': 1}
        self.end(200, json.dumps(res))

    def cmd_poke(self, post):
        STATE['responses'][post['pattern']] = post['text']

    def cmd_clear(self, post):
        STATE['responses'] = {}
        STATE['capture'] = None

    def cmd_begin_capture(self, post):
        STATE['capture'] = []

    def cmd_end_capture(self, post):
        ls = list(STATE['capture'] or [])
        STATE['capture'] = None
        return ls

    def end(self, status, res):
        if isinstance(res, str):
            res = res.encode('utf8')
        self.send_response(status)
        self.send_header("Content-Length", str(len(res)))
        self.end_headers()
        self.wfile.write(res)


server_address = '0.0.0.0', 8080

HTTPRequestHandler.protocol_version = 'HTTP/1.1'
httpd = http.server.HTTPServer(server_address, HTTPRequestHandler)

sa = httpd.socket.getsockname()
httpd.serve_forever()
