"""WSGI application for the web server."""

import gws
import gws.base.web
import gws.config

_STATE = {
    'inited': False,
}


def application(environ, start_response):
    """WSGI application, loads the configuration on the first call.

    Args:
        environ: WSGI environment.
        start_response: WSGI ``start_response`` function.

    Returns:
        The WSGI response iterable.
    """
    if not _STATE['inited']:
        init()
    root = gws.config.get_root()
    responder = handle_request(root, environ)
    return responder.send_response(environ, start_response)


def make_application(root):
    """Create a WSGI application for a given object tree root, without loading the configuration.

    Args:
        root: Object tree root.

    Returns:
        A WSGI application function.
    """
    def fn(environ, start_response):
        responder = handle_request(root, environ)
        return responder.send_response(environ, start_response)

    return fn


def init():
    """Load the configuration and set the log level.

    Exits the process if the configuration cannot be loaded.
    """
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
    """Reload the configuration."""
    _STATE['inited'] = False
    init()


def handle_request(root: gws.Root, environ) -> gws.WebResponder:
    """Handle a web request.

    Creates a requester for the application web site, parses the request and runs
    the middleware and the action. Errors are converted to error responses.

    Args:
        root: Object tree root.
        environ: WSGI environment.

    Returns:
        A responder.
    """
    req = gws.base.web.wsgi.Requester(root, environ, root.app.webMgr.site)

    try:
        req.parse()
    except Exception as exc:
        return handle_error(req, exc)

    gws.log.if_debug(_debug_repr, f'REQUEST_BEGIN <{req.method}> {req.command()}', req.params() or req.struct())
    gws.debug.time_start(f'REQUEST <{req.method}> {req.command()}')
    res = apply_middleware(root, req)
    gws.debug.time_end()
    gws.log.if_debug(_debug_repr, f'REQUEST_END <{req.method}> {req.command()}', res)

    return res


def apply_middleware(root: gws.Root, req: gws.WebRequester) -> gws.WebResponder:
    """Run the middleware and the action for a request.

    The ``enter_middleware`` methods are called in order until one of them returns a
    response. If none does, GET and POST requests are passed to the action, HEAD and
    OPTIONS requests get an empty response, other methods are not allowed. Then
    ``exit_middleware`` is called, in reverse order, for each middleware that was entered.

    Args:
        root: Object tree root.
        req: Web requester.

    Returns:
        A responder.
    """
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
            m = req.method
            if m == gws.RequestMethod.GET or m == gws.RequestMethod.POST:
                res = handle_action(root, req)
            elif m == gws.RequestMethod.HEAD or m == gws.RequestMethod.OPTIONS:
                res = req.content_responder(gws.ContentResponse(mimeType='text/plain', content=''))
            else:
                raise gws.base.web.error.MethodNotAllowed(['GET', 'POST', 'HEAD', 'OPTIONS'])
        except Exception as exc:
            res = handle_error(req, exc)

    for obj in reversed(done):
        try:
            obj.exit_middleware(req, res)
        except Exception as exc:
            res = handle_error(req, exc)

    return res


def _debug_repr(prefix, s):
    """Return a prefixed, truncated representation of an object for the debug log."""
    s = repr(gws.u.to_dict(s))
    m = 400
    n = len(s)
    if n <= m:
        return prefix + ': ' + s
    return prefix + ': ' + s[:m] + ' [...' + str(n - m) + ' more]'


def handle_error(req: gws.WebRequester, exc: Exception) -> gws.WebResponder:
    """Create an error response for an exception.

    Args:
        req: Web requester.
        exc: An exception, converted to an HTTP exception.

    Returns:
        A responder.
    """
    gws.log.if_debug(_debug_repr, f'REQUEST_ERROR', exc)
    web_exc = gws.base.web.error.from_exception(exc)
    return handle_http_error(req, web_exc)


def handle_http_error(req: gws.WebRequester, exc: gws.base.web.error.HTTPException) -> gws.WebResponder:
    """Create an error response for an HTTP exception.

    API requests get a structured response with the error code. Other requests get
    the site error page or the ``application.error`` template, rendered with the
    error code, or a plain error response if there is no template.

    Args:
        req: Web requester.
        exc: HTTP exception.

    Returns:
        A responder.
    """
    #
    # @TODO: image errors

    if req.isApi:
        return req.api_responder(
            gws.Response(
                status=exc.code,
                error=gws.ResponseError(
                    code=exc.code,
                    info=gws.u.get(exc, 'description', ''),
                ),
            )
        )

    error_template = getattr(req.site, 'errorPage', None)
    if not error_template:
        error_template = req.root.app.templateMgr.find_template('application.error', where=[])
    if not error_template:
        return req.error_responder(exc)

    args = gws.TemplateArgs(req=req, user=req.user, error=exc.code, status=exc.code or 500)
    res = error_template.render(gws.TemplateRenderInput(args=args))
    res.status = exc.code or 500
    return req.content_responder(res)


_relaxed_read_options = {
    gws.SpecReadOption.caseInsensitive,
    gws.SpecReadOption.convertValues,
    gws.SpecReadOption.ignoreExtraProps,
}


def handle_action(root: gws.Root, req: gws.WebRequester) -> gws.WebResponder:
    """Run the action command for a request.

    API requests use the structured payload as parameters. GET and POST requests use
    the GET parameters, which are read in a relaxed mode: case-insensitive, with value
    conversion and extra parameters ignored.

    Args:
        root: Object tree root.
        req: Web requester.

    Returns:
        A content, redirect or API responder, depending on the type of the command response.

    Raises:
        ``gws.NotFoundError``: If there is no command, or the command returns nothing.
        ``gws.base.web.error.MethodNotAllowed``: If the request method is not supported.
    """
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
        req.path(),
        req.user,
        read_options,
    )

    response = fn(req, request)

    if response is None:
        raise gws.NotFoundError(f'action not handled {category!r}:{req.command()!r}')

    if isinstance(response, gws.ContentResponse):
        return req.content_responder(response)

    if isinstance(response, gws.RedirectResponse):
        return req.redirect_responder(response)

    return req.api_responder(response)
