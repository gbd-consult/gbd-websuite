"""web application root"""

import gws
import gws.types as t
import gws.config
import gws.base.web.error
import gws.spec.runtime
import gws.base.web.site

from gws.base.auth.wsgi import WebRequest, WebResponse

_inited = False


def application(environ, start_response):
    global _inited

    if not _inited:
        _init()
        _inited = True

    response = handle_request(environ)
    return response(environ, start_response)


def reload():
    global _inited
    _inited = False


def handle_request(environ) -> WebResponse:
    root = gws.config.root()
    req = WebRequest(root, environ, _find_site(environ, root))
    return _handle_request2(req)


##


def _init():
    try:
        gws.log.info('initializing WEB application')
        root = gws.config.load()
        gws.log.set_level(root.application.var('server.log.level'))
    except:
        gws.log.exception('UNABLE TO LOAD CONFIGURATION')
        gws.exit(255)


##

class _DispatchError(gws.Error):
    pass


def _handle_request2(req: WebRequest) -> WebResponse:
    site = t.cast(gws.base.web.site.Object, req.site)

    cors = site.cors_options
    if cors and req.method == 'OPTIONS':
        return _with_cors_headers(cors, req.content_response(gws.ContentResponse(content='', mime='text/plain')))

    req.auth_open()
    res = _handle_request3(req)
    req.auth_close(res)

    if cors and req.method == 'POST':
        res = _with_cors_headers(cors, res)

    return res


def _handle_request3(req: WebRequest):
    try:
        req.parse_input()
        return _handle_action(req)
    except gws.base.web.error.HTTPException as err:
        return _handle_error(req, err)
    except:
        gws.log.exception()
        return _handle_error(req, gws.base.web.error.InternalServerError())


def _handle_error(req: WebRequest, err: gws.base.web.error.HTTPException) -> WebResponse:
    # @TODO: image errors

    site = t.cast(gws.base.web.site.Object, req.site)

    if req.output_struct_type:
        return req.struct_response(gws.Response(
            error=gws.ResponseError(
                status=err.code,
                info=gws.get(err, 'description', ''))))

    if not site.error_page:
        return req.error_response(err)

    try:
        r = site.error_page.render({
            'request': req,
            'error': err.code
        })
        return req.content_response(gws.ContentResponse(
            content=r.content,
            mime=r.mime,
            status=err.code))
    except:
        gws.log.exception()
        return req.error_response(gws.base.web.error.InternalServerError())


def _handle_action(req: WebRequest) -> WebResponse:
    cmd = req.param('cmd')
    if not cmd:
        raise gws.base.web.error.NotFound()

    if req.input_struct_type:
        method = 'api'
        params = req.params.get('params')
        strict = True
    elif req.method == 'GET':
        method = 'get'
        params = req.params
        strict = False
    elif req.method == 'POST':
        method = 'post'
        params = req.params
        strict = False
    else:
        # @TODO: add HEAD
        raise gws.base.web.error.MethodNotAllowed()

    try:
        command_desc = req.root.specs.check_command(cmd, method, params, with_strict_mode=strict)
    except gws.spec.runtime.Error as e:
        gws.log.error('ACTION ERROR', e)
        raise gws.base.web.error.BadRequest()

    if not command_desc:
        gws.log.error(f'command not found cmd={cmd!r} method={method!r}')
        raise gws.base.web.error.NotFound()

    project_uid = command_desc.params.get('projectUid')

    gws.log.debug(f'DISPATCH c={command_desc.cmd_name!r} a={command_desc.cmd_action!r} f={command_desc.function_name!r} projectUid={project_uid!r}')

    action = req.root.application.find_action(command_desc.cmd_action, project_uid)

    if not action:
        gws.log.error(f'action not found a={command_desc.cmd_action!r} method={method!r}')
        raise gws.base.web.error.NotFound()

    if not req.user.can_use(action):
        gws.log.error(f'permission denied a={command_desc.cmd_action!r} method={method!r}')
        raise gws.base.web.error.Forbidden()

    res = getattr(action, command_desc.function_name)(req, command_desc.params)

    if res is None:
        gws.log.error(f'action not handled cmd={cmd!r} method={method!r}')
        raise gws.base.web.error.NotFound()

    if isinstance(res, gws.ContentResponse):
        return req.content_response(res)

    return req.struct_response(res)


def _with_cors_headers(cors, res):
    if cors.get('allow_origin'):
        res.add_header('Access-Control-Allow-Origin', cors.get('allow_origin'))
    if cors.get('allow_credentials'):
        res.add_header('Access-Control-Allow-Credentials', 'true')
    if cors.get('allow_headers'):
        res.add_header('Access-Control-Allow-Headers', ', '.join(cors.get('allow_headers')))
    res.add_header('Access-Control-Allow-Methods', 'POST, OPTIONS')

    return res


def _find_site(environ, root):
    host = environ.get('HTTP_HOST', '').lower().split(':')[0].strip()

    for s in root.application.web_sites:
        if s.host.lower() == host:
            return s
    for s in root.application.web_sites:
        if s.host == '*':
            return s

    # there must be a '*' site (see application.config)
    raise ValueError('unknown host', host)
