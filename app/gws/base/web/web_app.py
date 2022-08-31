"""web application root"""

import gws
import gws.base.action
import gws.base.web.error
import gws.base.web.site
import gws.config
import gws.spec.runtime
import gws.types as t

from gws.base.web.wsgi import Requester, Responder

_STATE = {
    'inited': False,
    'middlewares': []
}


def application(environ, start_response):
    if not _STATE['inited']:
        _init()
    responder = handle_request(environ)
    return responder.send(environ, start_response)


def reload():
    _STATE['inited'] = False


def handle_request(environ) -> Responder:
    root = gws.config.root()
    req = gws.base.web.wsgi.Requester(root, environ, root.app.webSiteCollection.site_from_environ(environ))
    res = _handle_request(req)
    return res


##


def _final_middleware(req: Requester, nxt) -> Responder:
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


##

def _apply_middleware(req, n):
    return _STATE['middlewares'][n](req, lambda: _apply_middleware(req, n + 1))


def _handle_request(req: Requester) -> Responder:
    try:
        try:
            req.parse_input()
            return _apply_middleware(req, 0)
        except gws.base.web.error.HTTPException as exc:
            return _handle_error(req, exc)
    except:
        gws.log.exception()
        return req.error_responder(gws.base.web.error.InternalServerError())


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

    args = {'request': req, 'error': exc.code}
    response = req.site.errorPage.render(gws.TemplateRenderInput(args=args))
    return req.content_responder(response.with_attrs(status=exc.code))


def _init():
    try:
        gws.log.info('initializing WEB application')
        root = gws.config.load()
        gws.log.set_level(root.app.var('server.log.level'))
        _STATE['middlewares'] = root.app.web_middleware_list() + [_final_middleware]
        _STATE['inited'] = True
    except:
        gws.log.exception('UNABLE TO LOAD CONFIGURATION')
        gws.exit(255)
