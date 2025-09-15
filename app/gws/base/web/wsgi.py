"""Basic WSGI request/response handling."""

import gzip
import io
import os
import werkzeug.utils
import werkzeug.wrappers
import werkzeug.wsgi

import gws
import gws.lib.jsonx
import gws.lib.mime
import gws.lib.vendor.umsgpack as umsgpack

from . import error


class Responder(gws.WebResponder):
    def __init__(self, **kwargs):
        if 'wz' in kwargs:
            self._wz = kwargs['wz']
        else:
            self._wz = werkzeug.wrappers.Response(**kwargs)
        self.response = kwargs.get('response')
        self.status = self._wz.status_code

    def __repr__(self):
        return f'<Responder {self._wz}>'

    def send_response(self, environ, start_response):
        return self._wz(environ, start_response)

    def set_cookie(self, key, value, **kwargs):
        self._wz.set_cookie(key, value, **kwargs)

    def delete_cookie(self, key, **kwargs):
        self._wz.delete_cookie(key, **kwargs)

    def add_header(self, key, value):
        self._wz.headers.add(key, value)

    def set_status(self, status):
        self._wz.status_code = int(status)


class Requester(gws.WebRequester):
    _struct_mime = {
        'json': 'application/json',
        'msgpack': 'application/msgpack',
    }

    def __init__(self, root: gws.Root, environ: dict, site: gws.WebSite, **kwargs):
        if 'wz' in kwargs:
            self._wz = kwargs['wz']
        else:
            self._wz = werkzeug.wrappers.Request(environ)

        # this is also set in nginx (see server/ini), but we need this for unzipping (see data() below)
        self._wz.max_content_length = int(root.app.cfg('server.web.maxRequestLength', default=1)) * 1024 * 1024

        self.root = root
        self.site = site

        self.environ = self._wz.environ
        self.method = self._wz.method.upper()
        self.isSecure = self._wz.is_secure

        self.session = root.app.authMgr.guestSession
        self.user = root.app.authMgr.guestUser

        self.isPost = self.method == 'POST'
        self.isGet = self.method == 'GET'

        self.contentType = gws.lib.mime.get(self.header('content-type')) or gws.lib.mime.BIN

        self.inputType = None
        if self.isPost:
            self.inputType = self._struct_type(self.header('content-type'))

        self.outputType = None
        if self.inputType:
            self.outputType = self._struct_type(self.header('accept')) or self.inputType

        self.isApi = self.inputType is not None

        self._parsed_params = {}
        self._parsed_params_lc = {}
        self._parsed_struct = {}
        self._parsed_command = ''
        self._parsed = False
        self._uid = gws.u.mstime()

        if self.root.app.developer_option('request.log_all'):
            u = {
                'method': self.method,
                'path': self._wz.path,
                'query': self._wz.query_string,
                'cookies': self._wz.cookies,
                'headers': self._wz.headers,
                'environ': self._wz.environ,
            }
            gws.u.write_debug_file(f'request_{self._uid}', ''.join(f'{k}={v!r}\n' for k, v in u.items()))

    def __repr__(self):
        return f'<Requester {self._wz}>'

    def params(self):
        self._parse()
        return self._parsed_params

    def struct(self):
        self._parse()
        return self._parsed_struct

    def command(self):
        self._parse()
        return self._parsed_command

    def data(self):
        if not self.isPost:
            return None

        data = self._wz.get_data(as_text=False, parse_form_data=False)

        if self.root.app.developer_option('request.log_all'):
            gws.u.write_debug_file(f'request_{self._uid}.data', data)

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

    def has_param(self, key):
        self._parse()
        return key.lower() in self._parsed_params_lc

    def param(self, key, default=''):
        return self._parsed_params_lc.get(key.lower(), default)

    def header(self, key, default=''):
        return self._wz.headers.get(key, default)

    def cookie(self, key, default=''):
        return self._wz.cookies.get(key, default)

    def content_responder(self, response):
        args: dict = {
            'mimetype': response.mime,
            'status': response.status or 200,
            'headers': {},
            'direct_passthrough': False,
        }

        attach_name = None

        if response.attachmentName:
            attach_name = response.attachmentName
        elif response.asAttachment:
            if response.contentPath:
                attach_name = os.path.basename(response.contentPath)
            elif response.mime:
                ext = gws.lib.mime.extension_for(response.mime)
                attach_name = 'download.' + (ext or 'bin')
            else:
                attach_name = 'download'

        if attach_name:
            args['headers']['Content-Disposition'] = f'attachment; filename="{attach_name}"'
            args['mimetype'] = args['mimetype'] or gws.lib.mime.for_path(attach_name)

        if response.contentPath:
            args['response'] = werkzeug.wsgi.wrap_file(self.environ, open(response.contentPath, 'rb'))
            args['headers']['Content-Length'] = str(os.path.getsize(response.contentPath))
            args['mimetype'] = args['mimetype'] or gws.lib.mime.for_path(response.contentPath)
            args['direct_passthrough'] = True
        else:
            args['response'] = response.content

        if response.headers:
            args['headers'].update(response.headers)

        return Responder(**args)

    def redirect_responder(self, response):
        wz = werkzeug.utils.redirect(response.location, response.status or 302)
        if response.headers:
            wz.headers.update(response.headers)
        return Responder(wz=wz)

    def api_responder(self, response):
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

    def url_for(self, path, **kwargs):
        return self.site.url_for(self, path, **kwargs)

    ##

    def set_session(self, sess):
        self.session = sess
        self.user = sess.user

    ##

    _cmd_param_name = 'cmd'

    def _parse(self):
        if self._parsed:
            return

        # the server only understands requests to /_ or /_/commandName
        # GET params can be given as query string or encoded in the path
        # like _/commandName/param1/value1/param2/value2 etc

        path = self._wz.path
        path_parts = None

        if path == gws.c.SERVER_ENDPOINT:
            # example.com/_
            # the cmd param is expected to be in the query string or json
            cmd = ''
        elif path.startswith(gws.c.SERVER_ENDPOINT + '/'):
            # example.com/_/someCommand
            # the cmd param is in the url
            path_parts = path.split('/')
            cmd = path_parts[2]
            path_parts = path_parts[3:]
        else:
            raise error.NotFound(f'invalid request path: {path!r}')

        if self.inputType:
            self._parsed_struct = self._decode_struct(self.inputType)
            self._parsed_command = cmd or self._parsed_struct.pop(self._cmd_param_name, '')
        else:
            d = dict(self._wz.args)
            if path_parts:
                for n in range(1, len(path_parts), 2):
                    d[path_parts[n - 1]] = path_parts[n]
            self._parsed_command = cmd or d.pop(self._cmd_param_name, '')
            self._parsed_params = d
            self._parsed_params_lc = {k.lower(): v for k, v in d.items()}

        self._parsed = True

    def _struct_type(self, header):
        if header:
            header = header.lower()
            if header.startswith(self._struct_mime['json']):
                return 'json'
            if header.startswith(self._struct_mime['msgpack']):
                return 'msgpack'

    def _encode_struct(self, data, typ):
        if typ == 'json':
            return gws.lib.jsonx.to_string(data)
        if typ == 'msgpack':
            return umsgpack.dumps(data, default=gws.u.to_dict)
        raise ValueError('invalid struct type')

    def _decode_struct(self, typ):
        if typ == 'json':
            try:
                data = gws.u.require(self.data())
                s = data.decode(encoding='utf-8', errors='strict')
                return gws.lib.jsonx.from_string(s)
            except (UnicodeDecodeError, gws.lib.jsonx.Error):
                gws.log.error('malformed json request')
                raise error.BadRequest()

        if typ == 'msgpack':
            try:
                data = gws.u.require(self.data())
                return umsgpack.loads(data)
            except (TypeError, umsgpack.UnpackException):
                gws.log.error('malformed msgpack request')
                raise error.BadRequest()

        gws.log.error('invalid struct type')
        raise error.BadRequest()
