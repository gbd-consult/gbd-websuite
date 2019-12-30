import os

import werkzeug.wrappers
from werkzeug.utils import cached_property
from werkzeug.wsgi import wrap_file

import gws
import gws.tools.net
import gws.tools.json2
import gws.tools.vendor.umsgpack as umsgpack
import gws.types as t

from . import error

_JSON = 1
_MSGPACK = 2

_struct_mime = {
    _JSON: 'application/json',
    _MSGPACK: 'application/msgpack',
}


#:export IResponse
class BaseResponse(werkzeug.wrappers.Response, t.IResponse):
    pass


#:export IBaseRequest
class BaseRequest(t.IBaseRequest):
    def __init__(self, root: t.IRootObject, environ: dict, site: t.IWebSite):
        self._wz = werkzeug.wrappers.Request(environ)
        # the actual limit is set in the nginx conf (see server/ini)
        self._wz.max_content_length = 1024 * 1024 * 1024
        self.root: t.IRootObject = root
        self.site: t.IWebSite = site
        self.method: str = self._wz.method
        self.headers: dict = self._wz.headers

    @property
    def environ(self) -> dict:
        return self._wz.environ

    @property
    def cookies(self) -> dict:
        return self._wz.cookies

    @cached_property
    def input_struct_type(self) -> int:
        if self.method == 'POST':
            ct = self.headers.get('content-type', '').lower()
            if ct.startswith(_struct_mime[_JSON]):
                return _JSON
            if ct.startswith(_struct_mime[_MSGPACK]):
                return _MSGPACK
        return 0

    @cached_property
    def output_struct_type(self) -> int:
        h = self.headers.get('accept', '').lower()
        if _struct_mime[_MSGPACK] in h:
            return _MSGPACK
        if _struct_mime[_JSON] in h:
            return _JSON
        return self.input_struct_type

    @property
    def data(self) -> t.Optional[bytes]:
        if self.method != 'POST':
            return None
        return self._wz.get_data(as_text=False, parse_form_data=False)

    @property
    def text_data(self) -> t.Optional[str]:
        if self.method != 'POST':
            return None

        charset = self.headers.get('charset', 'utf-8')
        try:
            return self.data.decode(encoding=charset, errors='strict')
        except UnicodeDecodeError as e:
            gws.log.error('post data decoding error')
            raise error.BadRequest() from e

    @cached_property
    def params(self) -> dict:
        if self.input_struct_type:
            return self._decode_struct(self.input_struct_type)

        args = {k: v for k, v in self._wz.args.items()}
        path = self._wz.path

        # the server only understands requests to /_/...
        # the params can be given as query string or encoded in the path
        # like _/cmd/command/layer/la/x/12/y/34 etc

        if path == gws.SERVER_ENDPOINT:
            return args
        if path.startswith(gws.SERVER_ENDPOINT):
            return gws.extend(_params_from_path(path), args)

        gws.log.error(f'invalid request path: {path!r}')
        raise error.BadRequest()

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
    def kparams(self) -> dict:
        return {k.lower(): v for k, v in self.params.items()}

    def env(self, key: str, default: str = None) -> str:
        return self._wz.environ.get(key, default)

    def param(self, key: str, default: str = None) -> str:
        return self.params.get(key, default)

    def kparam(self, key: str, default: str = None) -> str:
        return self.kparams.get(key.lower(), default)

    def url_for(self, url: t.Url) -> t.Url:
        u = self.site.url_for(self, url)
        gws.log.debug(f'url_for: {url!r}=>{u!r}')
        return u

    def response(self, content: str, mimetype: str, status: int = 200) -> t.IResponse:
        r: t.IResponse = BaseResponse(
            content,
            mimetype=mimetype,
            status=status
        )
        return r

    def file_response(self, path: str, mimetype: str, status: int = 200, attachment_name: str = None) -> t.IResponse:
        headers = {
            'Content-Length': os.path.getsize(path)
        }
        if attachment_name:
            headers['Content-Disposition'] = f'attachment; filename="{attachment_name}"'

        fp = wrap_file(self.environ, open(path, 'rb'))

        r: t.IResponse = BaseResponse(
            fp,
            mimetype=mimetype,
            status=status,
            headers=headers,
            direct_passthrough=True
        )
        return r

    def struct_response(self, data: t.Response, status: int = 200) -> t.IResponse:
        typ = self.output_struct_type or _JSON
        return self.response(self._encode_struct(data, typ), _struct_mime[typ], status)

    def _encode_struct(self, data, typ):
        if typ == _JSON:
            return gws.tools.json2.to_string(data, pretty=True)
        if typ == _MSGPACK:
            return umsgpack.dumps(data, default=gws.as_dict)
        raise ValueError('invalid struct type')

    def _decode_struct(self, typ):
        if typ == _JSON:
            try:
                s = self.data.decode(encoding='utf-8', errors='strict')
                return gws.tools.json2.from_string(s)
            except (UnicodeDecodeError, gws.tools.json2.Error):
                gws.log.error('malformed json request')
                raise error.BadRequest()

        if typ == _MSGPACK:
            try:
                return umsgpack.loads(self.data)
            except (TypeError, umsgpack.UnpackException):
                gws.log.error('malformed msgpack request')
                raise error.BadRequest()

        raise ValueError('invalid struct type')


def _params_from_path(path):
    path = path.split('/')
    d = {}
    for n in range(2, len(path)):
        if n % 2:
            d[path[n - 1]] = path[n]
    return d
