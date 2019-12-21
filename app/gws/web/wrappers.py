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


Response = werkzeug.wrappers.Response

class Request:
    def __init__(self, root: t.RootObject, environ: dict, site: t.WebSiteObject):
        self.wz = werkzeug.wrappers.Request(environ)
        # the actual limit is set in the nginx conf (see server/ini)
        self.wz.max_content_length = 1024 * 1024 * 1024
        self.root = root
        self.site = site
        self.method = self.wz.method
        self.headers = self.wz.headers

    def response(self, content, mimetype, status=200):
        return Response(
            content,
            mimetype=mimetype,
            status=status)

    def struct_response(self, data, status=200):
        typ = self.expected_struct or _JSON

        if typ == _JSON:
            content = gws.tools.json2.to_string(data, pretty=True)
        elif typ == _MSGPACK:
            content = gws.tools.umsgpack.dumps(data, default=gws.as_dict)
        else:
            raise ValueError('invalid struct type')

        return self.response(content, _struct_mime[typ], status)

    @property
    def environ(self):
        return self.wz.environ

    @property
    def cookies(self):
        return self.wz.cookies

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
    def expected_struct(self):
        h = self.headers.get('accept', '').lower()
        if _struct_mime[_MSGPACK] in h:
            return _MSGPACK
        if _struct_mime[_JSON] in h:
            return _JSON
        if self._is_msgpack:
            return _MSGPACK
        if self._is_json:
            return _JSON

    @property
    def data(self):
        return self.wz.get_data(as_text=False, parse_form_data=False)

    def _decode_struct(self):
        if self._is_json:
            try:
                s = self.data.decode(encoding='utf-8', errors='strict')
                return gws.tools.json2.from_string(s)
            except (UnicodeDecodeError, gws.tools.json2.Error):
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
    def params(self):
        if self.has_struct:
            return self._decode_struct()

        args = {k: v for k, v in self.wz.args.items()}
        path = self.wz.path

        # the server only understands requests to /_/...
        # the params can be given as query string or encoded in the path
        # like _/cmd/command/layer/la/x/12/y/34 etc

        if path == gws.SERVER_ENDPOINT:
            return args
        if path.startswith(gws.SERVER_ENDPOINT):
            return gws.extend(_params_from_path(path), args)
        gws.log.error(f'invalid request path: {path!r}')
        return None

    @cached_property
    def post_data(self):
        if self.method != 'POST':
            return None

        charset = self.headers.get('charset', 'utf-8')
        try:
            return self.data.decode(encoding=charset, errors='strict')
        except UnicodeDecodeError:
            gws.log.error('post data decoding error')
            return None

    @cached_property
    def kparams(self):
        return {k.lower(): v for k, v in self.params.items()}

    def env(self, key, default=None):
        return self.wz.environ.get(key, default)

    def param(self, key, default=None):
        return self.params.get(key, default)

    def kparam(self, key, default=None):
        return self.kparams.get(key.lower(), default)

    def url_for(self, url):
        u = self.site.url_for(self, url)
        gws.log.debug(f'url_for: {url!r}=>{u!r}')
        return u


def _params_from_path(path):
    path = path.split('/')
    d = {}
    for n in range(2, len(path)):
        if n % 2:
            d[path[n - 1]] = path[n]
    return d
