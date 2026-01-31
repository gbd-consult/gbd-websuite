"""Basic WSGI request/response handling."""

import gzip
import io
import os
from typing import cast

import werkzeug.formparser
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
        self.status = self._wz.status_code

    def set_body(self, body):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self._wz.set_data(body)


class Requester(gws.WebRequester):
    _STRUCT_JSON = 'json'
    _STRUCT_MSGPACK = 'msgpack'

    _struct_mime = {
        _STRUCT_JSON: 'application/json',
        _STRUCT_MSGPACK: 'application/msgpack',
    }

    def __init__(self, root: gws.Root, environ: dict, site: gws.WebSite, **kwargs):
        if 'wz' in kwargs:
            self._wz = kwargs['wz']
        else:
            self._wz = werkzeug.wrappers.Request(environ)

        # this is also set in nginx (see server/ini), but we need this for unzipping (see data() below)
        self.maxContentLength = int(root.app.cfg('server.web.maxRequestLength') or 1) * 1024 * 1024
        self._wz.max_content_length = self.maxContentLength

        self.root = root
        self.site = site

        self.environ = self._wz.environ
        self.method = cast(gws.RequestMethod, self._wz.method.upper())
        self.isSecure = self._wz.is_secure

        self.session = root.app.authMgr.guestSession
        self.user = root.app.authMgr.guestUser

        self.isGet = self.method == gws.RequestMethod.GET
        self.isPost = self.method == gws.RequestMethod.POST
        self.isForm = False
        self.isApi = False

        self.structInput = None
        self.structOutput = None

        self.contentTypeHeader = self.header('content-type', '').lower().split(';')[0].strip()
        self.contentType = gws.lib.mime.get(self.contentTypeHeader) or gws.lib.mime.BIN

        if self.isPost:
            if self.contentTypeHeader == 'application/x-www-form-urlencoded' or self.contentTypeHeader == 'multipart/form-data':
                self.isForm = True
            else:
                self.structInput = self._struct_type(self.contentTypeHeader)
                if self.structInput:
                    self.isApi = True
                    self.structOutput = self._struct_type(self.header('accept')) or self.structInput

        self._parsed_params = {}
        self._parsed_params_lc = {}
        self._parsed_query_params = {}
        self._parsed_struct = {}
        self._parsed_command = ''
        self._parsed_path = ''
        self._parsed = False
        self._raw_post_data = None
        self._uid = gws.u.mstime()

        if self.root.app.developer_option('request.log_all'):
            u = {
                'method': self.method,
                'path': self._wz.path,
                'query': self._wz.query_string,
                'headers': self._wz.headers,
                'environ': self._wz.environ,
            }
            gws.u.write_debug_file(f'request_{self._uid}', ''.join(f'{k}={v!r}\n' for k, v in u.items()))

    def __repr__(self):
        return f'<Requester {self._wz}>'

    def parse(self):
        self._parse()

    def params(self):
        self._parse()
        return self._parsed_params

    def query_params(self):
        self._parse()
        return self._parsed_query_params

    def path(self):
        self._parse()
        return self._parsed_path

    def struct(self):
        self._parse()
        return self._parsed_struct

    def command(self):
        self._parse()
        return self._parsed_command

    def data(self):
        if not self.isPost:
            return b''

        if self._raw_post_data is not None:
            return self._raw_post_data

        cl = self.header('content-length')
        if not cl:
            self._raw_post_data = b''
            return self._raw_post_data
        try:
            cl = int(cl)
        except ValueError as exc:
            raise error.BadRequest('invalid content-length header') from exc
        if cl == 0:
            self._raw_post_data = b''
            return self._raw_post_data
        if cl > self.maxContentLength:
            raise error.RequestEntityTooLarge(f'content-length header too large: {cl}')

        data = self._wz.get_data(as_text=False, cache=False, parse_form_data=False)

        if self.root.app.developer_option('request.log_all'):
            gws.u.write_debug_file(f'request_{self._uid}.data', data)

        if self.header('content-encoding') == 'gzip':
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as fp:
                    data = fp.read(self.maxContentLength)
            except OSError as exc:
                raise error.BadRequest('gzip data error') from exc

        self._raw_post_data = data
        return data

    def text(self):
        data = self.data()
        if not data:
            return ''

        charset = self._wz.mimetype_params.get('charset', 'utf-8')
        try:
            return data.decode(encoding=charset, errors='strict')
        except UnicodeDecodeError as exc:
            raise error.BadRequest('text data decoding error') from exc

    def form(self):
        if not self.isForm:
            return []
        data = self.data()
        if not data:
            return []

        try:
            stream = io.BytesIO(data)
            opts = self._wz.mimetype_params

            # Fix for Qt multipart/form-data boundaries
            # Qt boundaries start with "boundary_.oOo._" and are base64-encoded
            # https://github.com/qt/qtbase/blob/04b7fc2de3c97174a725bbd4fdc0f6e496c85861/src/network/access/qhttpmultipart.cpp#L400
            # Werkzeug header parser does not understand base64 chars "/+="
            if data.startswith(b'--boundary_.oOo._'):
                boundary = data[2 : data.find(b'\r\n')].decode('ascii')
                opts['boundary'] = boundary

            parser = werkzeug.formparser.FormDataParser(silent=False)
            _, form, files = parser.parse(
                stream,
                mimetype=self.contentTypeHeader,
                content_length=len(data),
                options=opts,
            )
            return list(form.items()) + list(files.items())
        except Exception as exc:
            raise error.BadRequest(f'form decode error: {exc}') from exc

    def env(self, key, default=''):
        return self._wz.environ.get(key, default)

    def has_param(self, key):
        self._parse()
        return key.lower() in self._parsed_params_lc

    def param(self, key, default=''):
        self._parse()
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
        typ = self.structOutput or self._STRUCT_JSON
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

    _CMD_PARAM_NAME = 'cmd'

    def _parse(self):
        if not self._parsed:
            self._parse2()
            self._parsed = True

    def _parse2(self):
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
            self._parsed_path = '/'.join(path_parts)
        else:
            raise error.NotFound(f'invalid request path: {path!r}')

        if self.structInput:
            self._parsed_struct = self._decode_struct(self.structInput)
            self._parsed_command = cmd or self._parsed_struct.pop(self._CMD_PARAM_NAME, '')
        else:
            d = dict(self._wz.args)
            if path_parts:
                for n in range(1, len(path_parts), 2):
                    d[path_parts[n - 1]] = path_parts[n]
            self._parsed_command = cmd or d.pop(self._CMD_PARAM_NAME, '')
            self._parsed_params = d
            self._parsed_params_lc = {k.lower(): v for k, v in d.items()}
            self._parsed_query_params = dict(self._wz.args)

    def _struct_type(self, header):
        if header:
            header = header.lower()
            if header.startswith(self._struct_mime[self._STRUCT_JSON]):
                return self._STRUCT_JSON
            if header.startswith(self._struct_mime[self._STRUCT_MSGPACK]):
                return self._STRUCT_MSGPACK

    def _encode_struct(self, data, typ):
        if typ == self._STRUCT_JSON:
            return gws.lib.jsonx.to_string(data)
        if typ == self._STRUCT_MSGPACK:
            return umsgpack.dumps(data, default=gws.u.to_dict)
        raise ValueError(f'invalid struct type {typ!r}')

    def _decode_struct(self, typ):
        if typ == self._STRUCT_JSON:
            try:
                data = gws.u.require(self.data())
                s = data.decode(encoding='utf-8', errors='strict')
                return gws.lib.jsonx.from_string(s)
            except Exception as exc:
                raise error.BadRequest('malformed json request') from exc

        if typ == self._STRUCT_MSGPACK:
            try:
                data = gws.u.require(self.data())
                return umsgpack.loads(data)
            except Exception as exc:
                raise error.BadRequest('malformed msgpack request') from exc

        raise ValueError(f'invalid struct type {typ!r}')
