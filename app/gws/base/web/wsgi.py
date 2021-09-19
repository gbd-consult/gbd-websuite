import gzip
import io
import os

import werkzeug.utils
import werkzeug.wrappers
import werkzeug.wsgi
from werkzeug.utils import cached_property

import gws
import gws.types as t
import gws.lib.date
import gws.lib.json2
import gws.lib.net
import gws.lib.mime
import gws.lib.vendor.umsgpack as umsgpack

from . import error

_JSON = 1
_MSGPACK = 2

_struct_mime = {
    _JSON: 'application/json',
    _MSGPACK: 'application/msgpack',
}


class WebResponse(gws.IWebResponse):
    def __init__(self, **kwargs):
        if 'wz' in kwargs:
            self._wz = kwargs['wz']
        else:
            self._wz = werkzeug.wrappers.Response(**kwargs)

    @property
    def status_code(self):
        return self._wz.status_code

    def __call__(self, environ, start_response):
        return self._wz(environ, start_response)

    def set_cookie(self, key, **kwargs):
        self._wz.set_cookie(key, **kwargs)

    def delete_cookie(self, key, **kwargs):
        self._wz.delete_cookie(key, **kwargs)

    def add_header(self, key, value):
        self._wz.headers.add(key, value)


class WebRequest(gws.IWebRequest):
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
            gws.write_file_b(f'{gws.VAR_DIR}/debug_request_{gws.lib.date.timestamp_msec()}', data)

        if self.header('content-encoding') == 'gzip':
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as fp:
                return fp.read(self._wz.max_content_length)

        return data

    @property
    def text(self) -> t.Optional[str]:
        if self.method != 'POST' or not self.data:
            return None

        charset = self.header('charset', 'utf-8')
        try:
            return self.data.decode(encoding=charset, errors='strict')
        except UnicodeDecodeError as e:
            gws.log.error('post data decoding error')
            raise error.BadRequest() from e

    @property
    def is_secure(self) -> bool:
        return self._wz.is_secure

    def __init__(self, root: gws.RootObject, environ: dict, site: gws.IWebSite):
        self._wz = werkzeug.wrappers.Request(environ)
        # this is also set in nginx (see server/ini), but we need this for unzipping (see data() below)
        self._wz.max_content_length = int(root.application.var('server.web.maxRequestLength', default=1)) * 1024 * 1024

        self.params: t.Dict[str, t.Any] = {}
        self._lower_params: t.Dict[str, t.Any] = {}

        self.root: gws.RootObject = root
        self.site: gws.IWebSite = site
        self.method: str = self._wz.method

    def parse_input(self):
        self.params = self._parse_params() or {}
        self._lower_params = {k.lower(): v for k, v in self.params.items()}

    def env(self, key: str, default: str = '') -> str:
        return self._wz.environ.get(key, default)

    def param(self, key: str, default: str = '') -> str:
        return self._lower_params.get(key.lower(), default)

    def has_param(self, key: str) -> bool:
        return key.lower() in self._lower_params

    def header(self, key: str, default: str = '') -> str:
        return self._wz.headers.get(key, default)

    def cookie(self, key: str, default: str = '') -> str:
        return self._wz.cookies.get(key, default)

    def url_for(self, url: gws.Url) -> gws.Url:
        u = self.site.url_for(self, url)
        # gws.log.debug(f'url_for: {url!r}=>{u!r}')
        return u

    def response_object(self, **kwargs):
        return WebResponse(**kwargs)

    def content_response(self, res: gws.ContentResponse) -> WebResponse:
        if res.location:
            return WebResponse(wz=werkzeug.utils.redirect(res.location, res.status or 302))

        args: t.Dict = {
            'response': res.content,
            'mimetype': res.mime,
            'status': res.status or 200,
            'headers': {},
            'direct_passthrough': False,
        }

        if res.attachment_name or res.as_attachment:
            if res.attachment_name:
                attachment_name = res.attachment_name
            elif res.path:
                attachment_name = os.path.basename(res.path)
            elif res.mime:
                attachment_name = 'download.' + gws.lib.mime.extension(res.mime)
            else:
                raise gws.Error('missing attachment_name or mime type')

            args['headers']['Content-Disposition'] = f'attachment; filename="{attachment_name}"'
            args['mimetype'] = args['mimetype'] or gws.lib.mime.for_path(attachment_name)

        if res.path:
            args['response'] = werkzeug.wsgi.wrap_file(self.environ, open(res.path, 'rb'))
            args['headers']['Content-Length'] = str(os.path.getsize(res.path))
            args['mimetype'] = args['mimetype'] or gws.lib.mime.for_path(res.path)
            args['direct_passthrough'] = True

        return self.response_object(**args)

    def struct_response(self, res: gws.Response) -> WebResponse:
        typ = self.output_struct_type or _JSON
        status = 200
        if res.error:
            status = res.error.status or 400
        return self.response_object(
            response=self._encode_struct(res, typ),
            mimetype=_struct_mime[typ],
            status=status,
        )

    def error_response(self, err: error.HTTPException) -> WebResponse:
        return self.response_object(wz=err.get_response(self._wz.environ))

    def _parse_params(self):
        # the server only understands requests to /_ or /_/commandName
        # GET params can be given as query string or encoded in the path
        # like _/commandName/param1/value1/param2/value2 etc

        path = self._wz.path
        path_parts = None

        if path == gws.SERVER_ENDPOINT:
            # example.com/_
            # the cmd param is expected to be in the query string or json
            cmd = None
        elif path.startswith(gws.SERVER_ENDPOINT + '/'):
            # example.com/_/someCommand
            # the cmd param is in the url
            # if 'cmd' is also in the query string or json, they must match!
            path_parts = path.split('/')
            cmd = path_parts[2]
            path_parts = path_parts[3:]
        else:
            gws.log.error(f'invalid request path: {path!r}')
            raise error.NotFound()

        if self.input_struct_type:
            args = self._decode_struct(self.input_struct_type)
        else:
            args = dict(self._wz.args)
            if path_parts:
                for n in range(1, len(path_parts), 2):
                    args[path_parts[n - 1]] = path_parts[n]

        if cmd:
            cmd2 = args.get('cmd')
            if cmd2 and cmd2 != cmd:
                gws.log.error(f'cmd params do not match: {cmd!r} {cmd2!r}')
                raise error.BadRequest()
            args['cmd'] = cmd

        return args

    def _encode_struct(self, data, typ):
        if typ == _JSON:
            return gws.lib.json2.to_string(data, pretty=True)
        if typ == _MSGPACK:
            return umsgpack.dumps(data, default=gws.as_dict)
        raise ValueError('invalid struct type')

    def _decode_struct(self, typ):
        if typ == _JSON:
            try:
                s = self.data.decode(encoding='utf-8', errors='strict')
                return gws.lib.json2.from_string(s)
            except (UnicodeDecodeError, gws.lib.json2.Error):
                gws.log.error('malformed json request')
                raise error.BadRequest()

        if typ == _MSGPACK:
            try:
                return umsgpack.loads(self.data)
            except (TypeError, umsgpack.UnpackException):
                gws.log.error('malformed msgpack request')
                raise error.BadRequest()

        gws.log.error('invalid struct type')
        raise error.BadRequest()
