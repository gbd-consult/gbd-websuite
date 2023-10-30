import gzip
import io
import os
import werkzeug.utils
import werkzeug.wrappers
import werkzeug.wsgi

import gws
import gws.lib.date
import gws.lib.jsonx
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
        self.response = kwargs.get('response')
        self.status = self._wz.status_code

    def __repr__(self):
        return f'<Responder {self._wz}>'

    def send_response(self, environ, start_response):
        return self._wz(environ, start_response)

    def set_cookie(self, key, **kwargs):
        self._wz.set_cookie(key, **kwargs)

    def delete_cookie(self, key, **kwargs):
        self._wz.delete_cookie(key, **kwargs)

    def add_header(self, key, value):
        self._wz.headers.add(key, value)

    def set_status(self, value):
        self._wz.status_code = int(value)


class Requester(gws.IWebRequester):
    _struct_mime = {
        'json': 'application/json',
        'msgpack': 'application/msgpack',
    }

    def __init__(self, root: gws.IRoot, environ: dict, site: gws.IWebSite):
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

        self.inputType = None
        if self.isPost:
            self.inputType = self._struct_type(self.header('content-type'))

        self.outputType = None
        if self.inputType:
            self.outputType = self._struct_type(self.header('accept')) or self.inputType

        self.isApi = self.inputType is not None

        self.params: dict[str, t.Any] = {}
        self.lowerParams: dict[str, t.Any] = {}
        self.command = ''

    def __repr__(self):
        return f'<Requester {self._wz}>'

    def data(self):
        if not self.isPost:
            return None

        data = self._wz.get_data(as_text=False, parse_form_data=False)

        if self.root.app.developer_option('request.log_all'):
            gws.write_file_b(gws.ensure_dir(f'{gws.VAR_DIR}/debug') + '/request_{gws.lib.date.timestamp_msec()}', data)

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
        self.command, self.params = self._parse_params()
        self.lowerParams = {k.lower(): v for k, v in self.params.items()}

    def content_responder(self, response):
        if response.location:
            return Responder(wz=werkzeug.utils.redirect(response.location, response.status or 302))

        args: dict = {
            'mimetype': response.mime,
            'status': response.status or 200,
            'headers': {},
            'direct_passthrough': False,
        }

        aname = None

        if response.attachmentName:
            aname = response.attachmentName
        elif response.asAttachment:
            if response.path:
                aname = os.path.basename(response.path)
            elif response.mime:
                ext = gws.lib.mime.extension_for(response.mime)
                aname = 'download.' + ext
            else:
                aname = 'download'

        if aname:
            args['headers']['Content-Disposition'] = f'attachment; filename="{aname}"'
            args['mimetype'] = args['mimetype'] or gws.lib.mime.for_path(aname)

        if response.path:
            args['response'] = werkzeug.wsgi.wrap_file(self.environ, open(response.path, 'rb'))
            args['headers']['Content-Length'] = str(os.path.getsize(response.path))
            args['mimetype'] = args['mimetype'] or gws.lib.mime.for_path(response.path)
            args['direct_passthrough'] = True
        else:
            args['response'] = response.content

        if response.headers:
            args['headers'].update(response.headers)

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

    def url_for(self, path, **params):
        return self.site.url_for(self, path, **params)

    ##

    def require(self, uid, classref):
        return self.user.require(uid, classref)

    def require_project(self, uid):
        return t.cast(gws.IProject, self.require(uid, gws.ext.object.project))

    def require_layer(self, uid):
        return t.cast(gws.ILayer, self.require(uid, gws.ext.object.layer))

    def acquire(self, uid, classref):
        return self.user.acquire(uid, classref)

    def set_session(self, sess):
        self.session = sess
        self.user = sess.user

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
            cmd = ''
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
            params = self._decode_struct(self.inputType)
        else:
            params = dict(self._wz.args)
            if path_parts:
                for n in range(1, len(path_parts), 2):
                    params[path_parts[n - 1]] = path_parts[n]

        cmd = cmd or params.get('cmd', '')
        return cmd, params

    def _struct_type(self, header):
        if header:
            header = header.lower()
            if header.startswith(self._struct_mime['json']):
                return 'json'
            if header.startswith(self._struct_mime['msgpack']):
                return 'msgpack'

    def _encode_struct(self, data, typ):
        if typ == 'json':
            return gws.lib.jsonx.to_string(data, pretty=True)
        if typ == 'msgpack':
            return umsgpack.dumps(data, default=gws.to_dict)
        raise ValueError('invalid struct type')

    def _decode_struct(self, typ):
        if typ == 'json':
            try:
                s = self.data().decode(encoding='utf-8', errors='strict')
                return gws.lib.jsonx.from_string(s)
            except (UnicodeDecodeError, gws.lib.jsonx.Error):
                gws.log.error('malformed json request')
                raise error.BadRequest()

        if typ == 'msgpack':
            try:
                return umsgpack.loads(self.data())
            except (TypeError, umsgpack.UnpackException):
                gws.log.error('malformed msgpack request')
                raise error.BadRequest()

        gws.log.error('invalid struct type')
        raise error.BadRequest()
