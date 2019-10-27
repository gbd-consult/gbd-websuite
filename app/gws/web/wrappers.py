import re

import werkzeug.wrappers
from werkzeug.utils import cached_property

import gws
import gws.tools.net
import gws.tools.json2
import gws.tools.umsgpack
import gws.types as t

from . import error

_JSON = 1
_MSGPACK = 2

_struct_mime = {
    _JSON: 'application/json',
    _MSGPACK: 'application/msgpack',
}


class Response(werkzeug.wrappers.Response):
    environ = []
    request = None

    def html(self, s, status=200):
        return self.raw(s, mimetype='text/html', status=status)

    def struct(self, s, status=200):
        acc = self.request.wants_struct or _JSON
        data = ''

        if acc == _JSON:
            data = gws.tools.json2.to_string(s, pretty=True)
        if acc == _MSGPACK:
            data = gws.tools.umsgpack.dumps(s, default=gws.as_dict)

        return self.raw(data, mimetype=_struct_mime[acc], status=status)

    def raw(self, s, mimetype, status=200):
        return self.__class__(s, headers=self.headers, mimetype=mimetype, status=status)


class Request(werkzeug.wrappers.Request):
    # the actual limit is set in the nginx conf (see server/ini)
    max_content_length = 1024 * 1024 * 1024

    @property
    def response(self):
        r = Response()
        r.environ = self.environ
        r.request = self
        return r

    @cached_property
    def _is_json(self):
        h = self.headers.get('content-type', '').lower()
        return self.method == 'POST' and h.startswith(_struct_mime[_JSON])

    @cached_property
    def _is_msgpack(self):
        h = self.headers.get('content-type', '').lower()
        return self.method == 'POST' and h.startswith(_struct_mime[_MSGPACK])

    @cached_property
    def has_struct(self):
        return self._is_json or self._is_msgpack

    @cached_property
    def wants_struct(self):
        h = self.headers.get('accept', '').lower()
        if _struct_mime[_MSGPACK] in h:
            return _MSGPACK
        if _struct_mime[_JSON] in h:
            return _JSON
        if self._is_msgpack:
            return _MSGPACK
        if self._is_json:
            return _JSON

    def _decode_struct(self):
        if self._is_json:
            try:
                return gws.tools.json2.from_string(self.data)
            except gws.tools.json2.Error:
                gws.log.error('malformed json request')
                raise error.BadRequest()
        if self._is_msgpack:
            try:
                return gws.tools.umsgpack.loads(self.data)
            except (TypeError, gws.tools.umsgpack.UnpackException):
                gws.log.error('malformed msgpack request')
                raise error.BadRequest()
        return {}

    @cached_property
    def site(self):
        return self.environ['gws.site']

    @cached_property
    def params(self):
        if self.has_struct:
            return self._decode_struct()

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

        if self.method == 'POST':
            return self.form

    @cached_property
    def kparams(self):
        if self.method != 'GET':
            return {}
        return {k.lower(): v for k, v in self.params.items()}

    def env(self, key, default=None):
        return self.environ.get(key, default)

    def param(self, key, default=None):
        return self.params.get(key, default)

    def kparam(self, key, default=None):
        return self.kparams.get(key.lower(), default)

    def reversed_url(self, query_string):
        return self.site.reversed_rewrite(self, query_string)


def _params_from_path(path):
    path = path.split('/')
    d = {}
    for n in range(2, len(path)):
        if n % 2:
            d[path[n - 1]] = path[n]
    return d
