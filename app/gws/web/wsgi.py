"""web application root"""

import gws
import gws.common.template
import gws.config.loader
import gws.web

import gws.types as t

gws.config.loader.load()


def application(environ, start_response):
    res = _handle_request(environ)
    return res(environ, start_response)


##

_DEFAULT_CMD = 'assetHttpGetPath'


def _handle_request(environ):
    root = gws.config.root()
    req = gws.web.AuthRequest(root, environ, _find_site(environ, root))
    try:
        return _handle_request2(root, req)
    except gws.web.error.HTTPException as err:
        return _handle_error(root, req, err)
    except:
        gws.log.exception()
        return _handle_error(root, req, gws.web.error.InternalServerError())


def _handle_request2(root, req: gws.web.AuthRequest) -> gws.web.Response:
    if req.params is None:
        raise gws.web.error.NotFound()

    cors = req.site.cors

    if cors and req.method == 'OPTIONS':
        return _with_cors_headers(cors, req.response('', 'text/plain'))

    req.auth_begin()

    ## gws.p('REQUEST', {'user': req.user, 'params': req.params})

    res = _handle_action(root, req)

    req.auth_commit(res)

    if cors and req.method == 'POST':
        res = _with_cors_headers(cors, res)

    return res


def _handle_error(root, req, err):
    # @TODO: image errors

    if req.expected_struct:
        return req.struct_response(
            {'error': {'status': err.code, 'info': ''}},
            status=err.code)

    if not req.site.error_page:
        return err

    try:
        r = req.site.error_page.render({
            'request': req,
            'error': err.code
        })
        return req.response(r.content, r.mimeType, err.code)
    except:
        gws.log.exception()
        return gws.web.error.InternalServerError()


def _handle_action(root, req):
    cmd = req.param('cmd', _DEFAULT_CMD)

    # @TODO: add HEAD
    if req.has_struct:
        category = 'api'
    elif req.method == 'GET':
        category = 'http_get'
    elif req.method == 'POST':
        category = 'http_post'
    else:
        raise gws.web.error.MethodNotAllowed()

    try:
        if category == 'api':
            action_type, method_name, payload = root.validate_action(category, cmd, req.params.get('params'))
        else:
            action_type, method_name, payload = root.validate_action(category, cmd, req.params)
    except gws.config.Error as e:
        gws.log.error('ACTION ERROR', e)
        raise gws.web.error.BadRequest()

    project_uid = payload.get('projectUid')

    ## gws.log.debug(f'DISPATCH a={action_type!r} m={method_name!r} projectUid={project_uid!r}')

    action = root.application.find_action(action_type, project_uid)

    if not action:
        gws.log.error('handler not found', cmd)
        raise gws.web.error.NotFound()

    if not req.user.can_use(action):
        gws.log.error('permission denied', cmd)
        raise gws.web.error.Forbidden()

    # method_name does exist on action (action.validate ensures that)
    r = getattr(action, method_name)(req, payload)

    # now, r is a types.Response object
    if r is None:
        gws.log.error('action not handled', cmd)
        raise gws.web.error.NotFound()

    if isinstance(r, t.HttpResponse):
        return req.response(r.content, r.mimeType, r.get('status', 200))

    return req.struct_response(r)


def _with_cors_headers(cors, res: gws.web.Response):
    if cors.get('allowOrigin'):
        res.headers.add('Access-Control-Allow-Origin', cors.get('allowOrigin'))
    if cors.get('allowCredentials'):
        res.headers.add('Access-Control-Allow-Credentials', 'true')
    if cors.get('allowHeaders'):
        res.headers.add('Access-Control-Allow-Headers', ', '.join(cors.get('allowHeaders')))
    res.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')

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
