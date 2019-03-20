import re

import werkzeug.wrappers
from werkzeug.utils import cached_property

import gws
import gws.tools.net
import gws.tools.json2 as json2
import gws.types as t

from . import error


class Response(werkzeug.wrappers.Response):
    environ = []

    def html(self, s):
        return self.content(s, mimetype='text/html')

    def json(self, s):
        return self.content(json2.to_string(s, pretty=True), mimetype='application/json')

    def content(self, s, mimetype, status=200):
        return self.__class__(s, headers=self.headers, mimetype=mimetype, status=status)


class Request(werkzeug.wrappers.Request):
    max_content_length = 1024 * 1024 * 4

    @property
    def response(self):
        r = Response()
        r.environ = self.environ
        return r

    @cached_property
    def is_json(self):
        return self.method == 'POST' and self.headers.get('content-type', '').startswith('application/json')

    @cached_property
    def json(self):
        if self.is_json:
            try:
                return json2.from_string(self.data)
            except json2.Error:
                gws.log.error('malformed json request')
                raise error.BadRequest()
        return {}

    @cached_property
    def site(self):
        return self.environ['gws.site']

    @cached_property
    def params(self):
        if self.method == 'GET':
            args = {k: v for k, v in self.args.items()}

            # the server only understands requests to /_/...
            # the params can be given as query string or encoded in path
            # like _/cmd/command/layer/la/x/12/y/34 etc
            if self.path == gws.SERVER_ENDPOINT:
                return args
            if self.path.startswith(gws.SERVER_ENDPOINT):
                return gws.extend(_params_from_path(self.path), args)
            gws.log.error('invalid request path', self.path)
            return None

        if self.is_json:
            return self.json

        if self.method == 'POST':
            return self.form

    def env(self, key, default=None):
        return self.environ.get(key, default)

    def param(self, key, default=None):
        return self.params.get(key, default)

    def reversed_url(self, query_string):
        return self.site.reversed_rewrite(self, query_string)


def _params_from_path(path):
    path = path.split('/')
    d = {}
    for n in range(2, len(path)):
        if n % 2:
            d[path[n - 1]] = path[n]
    return d
