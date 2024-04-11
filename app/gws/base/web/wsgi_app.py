"""web application root"""

import gws
import gws.base.action
import gws.base.web
import gws.config
import gws.spec.runtime

_STATE = {
    'inited': False,
}


def application(environ, start_response):
    if not _STATE['inited']:
        init()
    responder = handle_request(environ)
    return responder.send_response(environ, start_response)


def initialized_application(environ, start_response):
    responder = handle_request(environ)
    return responder.send_response(environ, start_response)


def init():
    try:
        gws.log.info('initializing WEB application')
        gws.log.set_level('DEBUG')
        root = gws.config.load()
        gws.log.set_level(root.app.cfg('server.log.level'))
        _STATE['inited'] = True
    except:
        gws.log.exception('UNABLE TO LOAD CONFIGURATION')
        gws.exit(1)


def reload():
    _STATE['inited'] = False


def handle_request(environ) -> gws.IWebResponder:
    root = gws.config.root()
    site = root.app.webMgr.site_from_environ(environ)
    req = gws.base.web.wsgi.Requester(root, environ, site)

    try:
        req.initialize()
    except Exception as exc:
        return handle_error(req, exc)

    gws.log.if_debug(_debug_repr, f'REQUEST_BEGIN {req.command}', req.params)
    gws.time_start(f'REQUEST {req.command}')
    res = apply_middleware(root, req)
    gws.time_end()
    gws.log.if_debug(_debug_repr, f'REQUEST_END {req.command}', res)

    return res


def apply_middleware(root: gws.IRoot, req: gws.IWebRequester) -> gws.IWebResponder:
    res = None
    done = []

    for obj in root.app.middlewareMgr.objects():
        try:
            res = obj.enter_middleware(req)
            done.append(obj)
        except Exception as exc:
            res = handle_error(req, exc)

        if res:
            break

    if not res:
        try:
            res = handle_action(root, req)
        except Exception as exc:
            res = handle_error(req, exc)

    for obj in reversed(done):
        try:
            obj.exit_middleware(req, res)
        except Exception as exc:
            res = handle_error(req, exc)

    return res


def _debug_repr(prefix, s):
    s = repr(gws.to_dict(s))
    m = 400
    n = len(s)
    if n <= m:
        return prefix + ': ' + s
    return prefix + ': ' + s[:m] + ' [...' + str(n - m) + ' more]'


def handle_error(req: gws.IWebRequester, exc: Exception) -> gws.IWebResponder:
    if isinstance(exc, gws.base.web.error.HTTPException):
        return handle_http_error(req, exc)

    web_exc = None

    # convert our generic errors to http errors

    if isinstance(exc, gws.NotFoundError):
        web_exc = gws.base.web.error.NotFound()
    elif isinstance(exc, gws.ForbiddenError):
        web_exc = gws.base.web.error.Forbidden()
    elif isinstance(exc, gws.BadRequestError):
        web_exc = gws.base.web.error.BadRequest()
    elif isinstance(exc, gws.ResponseTooLargeError):
        web_exc = gws.base.web.error.Conflict()

    if web_exc:
        web_exc.__cause__ = exc
        return handle_http_error(req, web_exc)

    gws.log.exception()
    return handle_http_error(req, gws.base.web.error.InternalServerError())


def handle_http_error(req: gws.IWebRequester, exc: gws.base.web.error.HTTPException) -> gws.IWebResponder:
    #
    # @TODO: image errors

    gws.log.warning(f'HTTPException: {exc.code} cause={exc.__cause__}')

    if req.isApi:
        return req.api_responder(gws.Response(
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


_relaxed_read_options = {
    gws.SpecReadOption.caseInsensitive,
    gws.SpecReadOption.convertValues,
    gws.SpecReadOption.ignoreExtraProps,
}


def handle_action(root: gws.IRoot, req: gws.IWebRequester) -> gws.IWebResponder:
    if not req.command:
        raise gws.base.web.error.NotFound()

    if req.isApi:
        category = gws.CommandCategory.api
        params = req.params
        read_options = None
    elif req.isGet:
        category = gws.CommandCategory.get
        params = req.params
        read_options = _relaxed_read_options
    elif req.isPost:
        category = gws.CommandCategory.post
        params = req.params
        read_options = _relaxed_read_options
    else:
        # @TODO: add HEAD
        raise gws.base.web.error.MethodNotAllowed()

    fn, request = root.app.actionMgr.prepare_action(
        category,
        req.command,
        params,
        req.user,
        read_options
    )

    response = fn(req, request)

    if response is None:
        gws.log.error(f'action not handled {category!r}:{req.command!r}')
        raise gws.base.web.error.NotFound()

    if isinstance(response, gws.ContentResponse):
        return req.content_responder(response)

    if isinstance(response, gws.RedirectResponse):
        return req.redirect_responder(response)

    return req.api_responder(response)
