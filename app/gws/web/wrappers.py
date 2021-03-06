import os
import gzip
import io

import werkzeug.utils
import werkzeug.wrappers
import werkzeug.wsgi

from werkzeug.utils import cached_property

import gws
import gws.tools.date
import gws.tools.json2
import gws.tools.net
import gws.tools.vendor.umsgpack as umsgpack
import gws.web.error

import gws.types as t

_JSON = 1
_MSGPACK = 2

_struct_mime = {
    _JSON: 'application/json',
    _MSGPACK: 'application/msgpack',
}


#:export IResponse
class BaseResponse(t.IResponse):
    def __init__(self, **kwargs):
        if 'wz' in kwargs:
            self._wz = kwargs['wz']
        else:
            self._wz = werkzeug.wrappers.Response(**kwargs)

    def __call__(self, environ, start_response):
        return self._wz(environ, start_response)

    def set_cookie(self, key, **kwargs):
        self._wz.set_cookie(key, **kwargs)

    def delete_cookie(self, key, **kwargs):
        self._wz.delete_cookie(key, **kwargs)

    def add_header(self, key, value):
        self._wz.headers.add(key, value)


#:export IBaseRequest
class BaseRequest(t.IBaseRequest):
    def __init__(self, root: t.IRootObject, environ: dict, site: t.IWebSite):
        self._wz = werkzeug.wrappers.Request(environ)
        # this is also set in nginx (see server/ini), but we need this for unzipping (see data() below)
        self._wz.max_content_length = root.var('server.web.maxRequestLength') * 1024 * 1024

        self.params = {}
        self._lower_params = {}

        self.root: t.IRootObject = root
        self.site: t.IWebSite = site
        self.method: str = self._wz.method

    def init(self):
        self.params = self._parse_params() or {}
        self._lower_params = {k.lower(): v for k, v in self.params.items()}

    @property
    def environ(self) -> dict:
        return self._wz.environ

    @cached_property
    def input_struct_type(self) -> int:
        if self.method == 'POST':
            ct = self.header('content-type', '').lower()
            if ct.startswith(_struct_mime[_JSON]):
                return _JSON
            if ct.startswith(_struct_mime[_MSGPACK]):
                return _MSGPACK
        return 0

    @cached_property
    def output_struct_type(self) -> int:
        h = self.header('accept', '').lower()
        if _struct_mime[_MSGPACK] in h:
            return _MSGPACK
        if _struct_mime[_JSON] in h:
            return _JSON
        return self.input_struct_type

    @property
    def data(self) -> t.Optional[bytes]:
        if self.method != 'POST':
            return None

        data = self._wz.get_data(as_text=False, parse_form_data=False)

        if self.root.application.developer_option('request.log_all'):
            gws.write_file_b(f'{gws.VAR_DIR}/debug_request_{gws.tools.date.timestamp_msec()}', data)

        if self.header('content-encoding') == 'gzip':
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as fp:
                return fp.read(self._wz.max_content_length)

        return data

    @property
    def text(self) -> t.Optional[str]:
        if self.method != 'POST':
            return None

        charset = self.header('charset', 'utf-8')
        try:
            return self.data.decode(encoding=charset, errors='strict')
        except UnicodeDecodeError as e:
            gws.log.error('post data decoding error')
            raise gws.web.error.BadRequest() from e

    @property
    def is_secure(self) -> bool:
        return self._wz.is_secure

    def env(self, key: str, default: str = None) -> str:
        return self._wz.environ.get(key, default)

    def param(self, key: str, default: str = None) -> str:
        return self._lower_params.get(key.lower(), default)

    def has_param(self, key: str) -> bool:
        return key.lower() in self._lower_params

    def header(self, key: str, default: str = None) -> str:
        return self._wz.headers.get(key, default)

    def cookie(self, key: str, default: str = None) -> str:
        return self._wz.cookies.get(key, default)

    def url_for(self, url: t.Url) -> t.Url:
        u = self.site.url_for(self, url)
        # gws.log.debug(f'url_for: {url!r}=>{u!r}')
        return u

    def response(self, content: str, mimetype: str, status: int = 200) -> t.IResponse:
        return BaseResponse(
            response=content,
            mimetype=mimetype,
            status=status
        )

    def redirect_response(self, location, status=302):
        return werkzeug.utils.redirect(location, status)

    def file_response(self, path: str, mimetype: str, status: int = 200, attachment_name: str = None) -> t.IResponse:
        headers = {
            'Content-Length': os.path.getsize(path)
        }
        if attachment_name:
            headers['Content-Disposition'] = f'attachment; filename="{attachment_name}"'

        fp = werkzeug.wsgi.wrap_file(self.environ, open(path, 'rb'))

        return BaseResponse(
            response=fp,
            mimetype=mimetype,
            status=status,
            headers=headers,
            direct_passthrough=True
        )

    def struct_response(self, data: t.Response, status: int = 200) -> t.IResponse:
        typ = self.output_struct_type or _JSON
        return self.response(self._encode_struct(data, typ), _struct_mime[typ], status)

    def error_response(self, err) -> t.IResponse:
        return BaseResponse(wz=err.get_response(self._wz.environ))

    def _parse_params(self):
        if self.input_struct_type:
            return self._decode_struct(self.input_struct_type)

        args = {k: v for k, v in self._wz.args.items()}
        path = self._wz.path

        # the server only understands requests to /_/...
        # the params can be given as query string or encoded in the path
        # like _/cmd/command/layer/la/x/12/y/34 etc

        if path == gws.SERVER_ENDPOINT:
            return args

        if path.startswith(gws.SERVER_ENDPOINT + '/'):
            p = path.split('/')
            for n in range(3, len(p), 2):
                args[p[n - 1]] = p[n]
            return args

        gws.log.error(f'invalid request path: {path!r}')
        raise gws.web.error.NotFound()

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
                raise gws.web.error.BadRequest()

        if typ == _MSGPACK:
            try:
                return umsgpack.loads(self.data)
            except (TypeError, umsgpack.UnpackException):
                gws.log.error('malformed msgpack request')
                raise gws.web.error.BadRequest()

        gws.log.error('invalid struct type')
        raise gws.web.error.BadRequest()
