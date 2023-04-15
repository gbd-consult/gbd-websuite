"""web application root"""

import gws
import gws.base.action
import gws.base.web.error
import gws.base.web.site
import gws.base.web.wsgi
import gws.config
import gws.spec.runtime

_STATE = {
    'inited': False,
    'middleware': []
}


def application(environ, start_response):
    if not _STATE['inited']:
        init()
    responder = handle_request(environ)
    return responder.send_response(environ, start_response)


def init():
    try:
        gws.log.info('initializing WEB application')
        root = gws.config.load()
        gws.log.set_level(root.app.cfg('server.log.level'))

        _STATE['middleware'] = root.app.web_middleware_list() + [final_middleware]
        _STATE['inited'] = True
    except:
        gws.log.exception('UNABLE TO LOAD CONFIGURATION')
        gws.exit(1)


def reload():
    _STATE['inited'] = False


def handle_request(environ) -> gws.IWebResponder:
    root = gws.config.root()
    site = root.app.webMgr.site_from_environ(environ)
    req = gws.base.web.wsgi.Requester(root, environ, site, _STATE['middleware'])

    try:
        try:
            req.parse_input()
            gws.log.if_debug(_debug_repr, 'REQUEST_BEGIN:', req.params)
            res = req.apply_middleware()
            gws.log.if_debug(_debug_repr, 'REQUEST_END:', res)
            return res
        except gws.base.web.error.HTTPException as exc:
            return handle_error(req, exc)
    except:
        gws.log.exception()
        return req.error_responder(gws.base.web.error.InternalServerError())


def _debug_repr(prefix, s):
    s = repr(gws.to_dict(s))
    m = 400
    n = len(s)
    if n <= m:
        return prefix + ': ' + s
    return prefix + ': ' + s[:m] + ' [...' + str(n - m) + ' more]'


def handle_error(req: gws.IWebRequester, exc: gws.base.web.error.HTTPException) -> gws.IWebResponder:
    # @TODO: image errors

    gws.log.warning(f'HTTPException: {exc.code}')

    if req.isApi:
        return req.struct_responder(gws.Response(
            status=exc.code,
            error=gws.ResponseError(
                code=exc.code,
                info=gws.get(exc, 'description', ''))))

    if not req.site.errorPage:
        return req.error_responder(exc)

    args = {'request': req, 'error': exc.code}
    response = req.site.errorPage.render(gws.TemplateRenderInput(args=args))
    response.status = exc.code
    return req.content_responder(response)


_relaxed_read_options = {'case_insensitive', 'convert_values', 'ignore_extra_props'}


def final_middleware(req: gws.IWebRequester, nxt) -> gws.IWebResponder:
    command_name = req.param('cmd')
    if not command_name:
        raise gws.base.web.error.NotFound()

    read_options = None

    if req.isApi:
        command_category = 'api'
        params = req.param('params')
    elif req.isGet:
        command_category = 'get'
        params = req.params
        read_options = _relaxed_read_options
    elif req.isPost:
        command_category = 'post'
        params = req.params
        read_options = _relaxed_read_options
    else:
        # @TODO: add HEAD
        raise gws.base.web.error.MethodNotAllowed()

    try:
        fn, request = gws.base.action.dispatch(
            req.root,
            command_category,
            command_name,
            params,
            req.user,
            read_options
        )
    except gws.base.action.CommandNotFound as exc:
        raise gws.base.web.error.NotFound() from exc
    except gws.base.action.CommandForbidden as exc:
        raise gws.base.web.error.Forbidden() from exc
    except gws.base.action.BadRequest as exc:
        raise gws.base.web.error.BadRequest() from exc

    response = fn(req, request)

    if response is None:
        gws.log.error(f'action not handled {command_category!r}:{command_name!r}')
        raise gws.base.web.error.NotFound()

    if isinstance(response, gws.ContentResponse):
        return req.content_responder(response)

    return req.struct_responder(response)
