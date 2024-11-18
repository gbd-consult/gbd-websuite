"""web application root"""

from typing import cast

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
    root = gws.config.root()
    responder = handle_request(root, environ)
    return responder.send_response(environ, start_response)


def make_application(root):
    def fn(environ, start_response):
        responder = handle_request(root, environ)
        return responder.send_response(environ, start_response)

    return fn


def init():
    try:
        gws.log.info('initializing WEB application')
        gws.log.set_level('DEBUG')
        root = gws.config.load()
        gws.log.set_level(root.app.cfg('server.log.level'))
        _STATE['inited'] = True
    except:
        gws.log.exception('UNABLE TO LOAD CONFIGURATION')
        gws.u.exit(1)


def reload():
    _STATE['inited'] = False
    init()


def handle_request(root: gws.Root, environ) -> gws.WebResponder:
    site = root.app.webMgr.site_from_environ(environ)
    req = gws.base.web.wsgi.Requester(root, environ, site)

    try:
        _ = req.params()  # enforce parsing
    except Exception as exc:
        return handle_error(req, exc)

    gws.log.if_debug(_debug_repr, f'REQUEST_BEGIN {req.command()}', req.params() or req.struct())
    gws.debug.time_start(f'REQUEST {req.command()}')
    res = apply_middleware(root, req)
    gws.debug.time_end()
    gws.log.if_debug(_debug_repr, f'REQUEST_END {req.command()}', res)

    return res


def apply_middleware(root: gws.Root, req: gws.WebRequester) -> gws.WebResponder:
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
    s = repr(gws.u.to_dict(s))
    m = 400
    n = len(s)
    if n <= m:
        return prefix + ': ' + s
    return prefix + ': ' + s[:m] + ' [...' + str(n - m) + ' more]'


def handle_error(req: gws.WebRequester, exc: Exception) -> gws.WebResponder:
    web_exc = gws.base.web.error.from_exception(exc)
    return handle_http_error(req, web_exc)


def handle_http_error(req: gws.WebRequester, exc: gws.base.web.error.HTTPException) -> gws.WebResponder:
    #
    # @TODO: image errors

    if req.isApi:
        return req.api_responder(gws.Response(
            status=exc.code,
            error=gws.ResponseError(
                code=exc.code,
                info=gws.u.get(exc, 'description', ''))))

    if not req.site.errorPage:
        return req.error_responder(exc)

    args = gws.TemplateArgs(
        req=req,
        user=req.user,
        error=exc.code
    )
    res = req.site.errorPage.render(gws.TemplateRenderInput(args=args))
    res.status = exc.code
    return req.content_responder(res)


_relaxed_read_options = {
    gws.SpecReadOption.caseInsensitive,
    gws.SpecReadOption.convertValues,
    gws.SpecReadOption.ignoreExtraProps,
}


def handle_action(root: gws.Root, req: gws.WebRequester) -> gws.WebResponder:
    if not req.command():
        raise gws.NotFoundError('no command provided')

    if req.isApi:
        category = gws.CommandCategory.api
        params = req.struct()
        read_options = None
    elif req.isGet:
        category = gws.CommandCategory.get
        params = req.params()
        read_options = _relaxed_read_options
    elif req.isPost:
        category = gws.CommandCategory.post
        params = req.params()
        read_options = _relaxed_read_options
    else:
        # @TODO: add HEAD
        raise gws.base.web.error.MethodNotAllowed()

    fn, request = root.app.actionMgr.prepare_action(
        category,
        req.command(),
        params,
        req.user,
        read_options
    )

    response = fn(req, request)

    if response is None:
        raise gws.NotFoundError(f'action not handled {category!r}:{req.command()!r}')

    if isinstance(response, gws.ContentResponse):
        return req.content_responder(response)

    if isinstance(response, gws.RedirectResponse):
        return req.redirect_responder(response)

    return req.api_responder(response)
