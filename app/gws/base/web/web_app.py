"""web application root"""

import gws
import gws.base.action
import gws.base.web.error
import gws.base.web.site
import gws.config
import gws.spec.runtime
import gws.types as t

from gws.base.web.wsgi import Requester, Responder

_inited = False


def application(environ, start_response):
    global _inited

    if not _inited:
        _init()
        _inited = True

    responder = handle_request(environ)
    return responder(environ, start_response)


def reload():
    global _inited
    _inited = False


def handle_request(environ) -> Responder:
    root = gws.config.root()
    req = gws.base.web.wsgi.Requester(root, environ, _find_site(root, environ))
    return _handle_request2(req)


##


def _init():
    try:
        gws.log.info('initializing WEB application')
        root = gws.config.load()
        gws.log.set_level(root.app.var('server.log.level'))
    except:
        gws.log.exception('UNABLE TO LOAD CONFIGURATION')
        gws.exit(255)


##

class _DispatchError(gws.Error):
    pass


def _handle_request2(req: Requester) -> Responder:
    site = t.cast(gws.base.web.site.Object, req.site)

    cors = site.corsOptions
    if cors and req.method == 'OPTIONS':
        return _with_cors_headers(cors, req.content_responder(gws.ContentResponse(content='', mime='text/plain')))

    try:
        res = _handle_request3(req)
    except:
        gws.log.exception()
        res = _handle_error(req, gws.base.web.error.InternalServerError())

    if cors:
        res = _with_cors_headers(cors, res)

    return res


def _handle_request3(req: Requester) -> Responder:
    try:
        req.parse_input()
    except gws.base.web.error.HTTPException as exc:
        return _handle_error(req, exc)

    req.enter_request()
    try:
        res = _handle_request4(req)
    except gws.base.web.error.HTTPException as exc:
        res = _handle_error(req, exc)
    req.exit_request(res)

    return res


def _handle_request4(req: Requester) -> Responder:
    command_name = req.param('cmd')
    if not command_name:
        raise gws.base.web.error.NotFound()

    if req.isApi:
        command_category = 'api'
        params = req.param('params')
        strict_mode = True
    elif req.isGet:
        command_category = 'get'
        params = req.params
        strict_mode = False
    elif req.isPost:
        command_category = 'post'
        params = req.params
        strict_mode = False
    else:
        # @TODO: add HEAD
        raise gws.base.web.error.MethodNotAllowed()

    command_desc = req.root.app.command_descriptor(
        command_category,
        command_name,
        params,
        req.user,
        strict_mode
    )

    response = command_desc.methodPtr(req, command_desc.request)

    if response is None:
        gws.log.error(f'action not handled {command_category!r}:{command_name!r}')
        raise gws.base.web.error.NotFound()

    if isinstance(response, gws.ContentResponse):
        return req.content_responder(response)

    return req.struct_responder(response)


def _handle_error(req: Requester, exc: gws.base.web.error.HTTPException) -> Responder:
    # @TODO: image errors

    if req.isApi:
        return req.struct_responder(gws.Response(
            status=exc.code,
            error=gws.ResponseError(
                status=exc.code,
                info=gws.get(exc, 'description', ''))))

    if not req.site.errorPage:
        return req.error_responder(exc)

    try:
        args = {'request': req, 'error': exc.code}
        response = req.site.errorPage.render(gws.TemplateRenderInput(args=args))
        return req.content_responder(response.with_attrs(status=exc.code))
    except:
        gws.log.exception()
        return req.error_responder(gws.base.web.error.InternalServerError())


def _with_cors_headers(cors, res):
    if cors.get('allow_origin'):
        res.add_header('Access-Control-Allow-Origin', cors.get('allow_origin'))
    if cors.get('allow_credentials'):
        res.add_header('Access-Control-Allow-Credentials', 'true')
    if cors.get('allow_headers'):
        res.add_header('Access-Control-Allow-Headers', ', '.join(cors.get('allow_headers')))
    res.add_header('Access-Control-Allow-Methods', 'POST, OPTIONS')

    return res


def _find_site(root, environ):
    host = environ.get('HTTP_HOST', '').lower().split(':')[0].strip()

    for s in root.app.webSites:
        if s.host.lower() == host:
            return s
    for s in root.app.webSites:
        if s.host == '*':
            return s

    # there must be a '*' site (see application.config)
    raise ValueError('unknown host', host)
