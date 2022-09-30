import gzip
import io
import os
import werkzeug.utils
import werkzeug.wrappers
import werkzeug.wsgi

import gws
import gws.lib.date
import gws.lib.json2
import gws.lib.mime
import gws.lib.vendor.umsgpack as umsgpack
import gws.types as t

from . import error


class Responder(gws.IWebResponder):
    def __init__(self, **kwargs):
        if 'wz' in kwargs:
            self._wz = kwargs['wz']
        else:
            self._wz = werkzeug.wrappers.Response(**kwargs)
        self.status = self._wz.status_code

    def send_response(self, environ, start_response):
        return self._wz(environ, start_response)

    def set_cookie(self, key, **kwargs):
        self._wz.set_cookie(key, **kwargs)

    def delete_cookie(self, key, **kwargs):
        self._wz.delete_cookie(key, **kwargs)

    def add_header(self, key, value):
        self._wz.headers.add(key, value)


class Requester(gws.IWebRequester):
    _struct_mime = {
        'json': 'application/json',
        'msgpack': 'application/msgpack',
    }

    _middlewareList: t.List[gws.WebMiddlewareHandler]
    _middlewareIndex: int

    def __init__(self, root: gws.IRoot, environ: dict, site: gws.IWebSite, middleware: t.List[gws.WebMiddlewareHandler]):
        self._wz = werkzeug.wrappers.Request(environ)
        # this is also set in nginx (see server/ini), but we need this for unzipping (see data() below)
        self._wz.max_content_length = int(root.app.var('server.web.maxRequestLength', default=1)) * 1024 * 1024

        self._middlewareList = middleware
        self._middlewareIndex = 0

        self.root = root
        self.site = site

        self.environ = self._wz.environ
        self.method = self._wz.method.upper()
        self.isSecure = self._wz.is_secure

        self.isPost = self.method == 'POST'
        self.isGet = self.method == 'GET'

        self.inputType = None
        if self.isPost:
            self.inputType = self._struct_type(self.header('content-type'))

        self.outputType = None
        if self.inputType:
            self.outputType = self._struct_type(self.header('accept')) or self.inputType

        self.isApi = self.inputType is not None

        self.params: t.Dict[str, t.Any] = {}
        self.lowerParams: t.Dict[str, t.Any] = {}

    def apply_middleware(self):
        fn = self._middlewareList[self._middlewareIndex]
        self._middlewareIndex += 1
        return fn(self, self.apply_middleware)

    def data(self):
        if not self.isPost:
            return None

        data = self._wz.get_data(as_text=False, parse_form_data=False)

        if self.root.app.developer_option('request.log_all'):
            gws.write_file_b(f'{gws.VAR_DIR}/debug_request_{gws.lib.date.timestamp_msec()}', data)

        if self.header('content-encoding') == 'gzip':
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as fp:
                return fp.read(self._wz.max_content_length)

        return data

    def text(self):
        data = self.data()
        if data is None:
            return None

        charset = self.header('charset', 'utf-8')
        try:
            return data.decode(encoding=charset, errors='strict')
        except UnicodeDecodeError as exc:
            gws.log.error('post data decoding error')
            raise error.BadRequest() from exc

    def env(self, key, default=''):
        return self._wz.environ.get(key, default)

    def param(self, key, default=''):
        return self.lowerParams.get(key.lower(), default)

    def has_param(self, key):
        return key.lower() in self.lowerParams

    def header(self, key, default=''):
        return self._wz.headers.get(key, default)

    def cookie(self, key, default=''):
        return self._wz.cookies.get(key, default)

    def parse_input(self):
        self.params = self._parse_params() or {}
        self.lowerParams = {k.lower(): v for k, v in self.params.items()}

    def content_responder(self, response):
        if response.location:
            return Responder(wz=werkzeug.utils.redirect(response.location, response.status or 302))

        args: t.Dict = {
            'mimetype': response.mime,
            'status': response.status or 200,
            'headers': {},
            'direct_passthrough': False,
        }

        def _attachment_name():
            if isinstance(response.attachment, str):
                return response.attachment
            if response.path:
                return os.path.basename(response.path)
            if response.mime:
                ext = gws.lib.mime.extension_for(response.mime)
                if ext:
                    return 'download.' + ext
            raise gws.Error('missing attachment name or mime type')

        if response.attachment:
            name = _attachment_name()
            args['headers']['Content-Disposition'] = f'attachment; filename="{name}"'
            args['mimetype'] = args['mimetype'] or gws.lib.mime.for_path(name)

        if response.path:
            args['response'] = werkzeug.wsgi.wrap_file(self.environ, open(response.path, 'rb'))
            args['headers']['Content-Length'] = str(os.path.getsize(response.path))
            args['mimetype'] = args['mimetype'] or gws.lib.mime.for_path(response.path)
            args['direct_passthrough'] = True
        elif response.text is not None:
            args['response'] = response.text
        else:
            args['response'] = response.content

        return Responder(**args)

    def struct_responder(self, response):
        typ = self.outputType or 'json'
        return Responder(
            response=self._encode_struct(response, typ),
            mimetype=self._struct_mime[typ],
            status=response.status or 200,
        )

    def error_responder(self, exc):
        err = exc if isinstance(exc, error.HTTPException) else error.InternalServerError()
        return Responder(wz=err.get_response(self._wz.environ))

    ##

    def require(self, classref, uid):
        obj = self.root.find(classref, uid)
        if obj and self.user and self.user.can_use(obj):
            return obj
        if not obj:
            gws.log.error('require: not found', classref, uid)
            raise gws.base.web.error.NotFound()
        gws.log.error('require: denied', classref, uid)
        raise gws.base.web.error.Forbidden()

    def require_project(self, uid):
        return t.cast(gws.IProject, self.require(gws.ext.object.project, uid))

    def require_layer(self, uid):
        return t.cast(gws.ILayer, self.require(gws.ext.object.layer, uid))

    def acquire(self, classref, uid):
        obj = self.root.find(classref, uid)
        if obj and self.user and self.user.can_use(obj):
            return obj

    ##

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

        if self.inputType:
            args = self._decode_struct(self.inputType)
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

    def _struct_type(self, header):
        if header:
            header = header.lower()
            if header.startswith(self._struct_mime['json']):
                return 'json'
            if header.startswith(self._struct_mime['msgpack']):
                return 'msgpack'

    def _encode_struct(self, data, typ):
        if typ == 'json':
            return gws.lib.json2.to_string(data, pretty=True)
        if typ == 'msgpack':
            return umsgpack.dumps(data, default=gws.to_dict)
        raise ValueError('invalid struct type')

    def _decode_struct(self, typ):
        if typ == 'json':
            try:
                s = self.data().decode(encoding='utf-8', errors='strict')
                return gws.lib.json2.from_string(s)
            except (UnicodeDecodeError, gws.lib.json2.Error):
                gws.log.error('malformed json request')
                raise error.BadRequest()

        if typ == 'msgpack':
            try:
                return umsgpack.loads(self.data)
            except (TypeError, umsgpack.UnpackException):
                gws.log.error('malformed msgpack request')
                raise error.BadRequest()

        gws.log.error('invalid struct type')
        raise error.BadRequest()